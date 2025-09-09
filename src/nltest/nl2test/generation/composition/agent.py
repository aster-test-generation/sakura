from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import json
from pathlib import Path

from langchain_core.messages import ToolMessage, ToolCall
from langchain_core.tools import BaseTool

from nltest.nl2test.core.react_agent import ReActAgent
from nltest.nl2test.models import AgentState, DecompositionMode
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.utils.llm.llm_client import LLMClient
from nltest.utils.file_io.test_file_manager import TestFileManager, TestFileInfo
from nltest.utils.exceptions import FileDeletionError
from nltest.utils.execution import JavaCompilation
from nltest.utils.execution.execution import JavaExecution


class CompositionReActAgent(ReActAgent):
    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: List[BaseTool],
        allow_duplicate_tools: List[BaseTool] | None = None,
        system_message: str,
        project_root: Path,
        max_iters: int = 30,
    ):
        # Allow caller to inject system_message for parity with localization

        super().__init__(
            llm=llm,
            tools=tools,
            allow_duplicate_tools=allow_duplicate_tools,
            system_message=system_message,
            allow_parallelize=True,
            max_iters=max_iters,
        )
        self.project_root = Path(project_root)

    def _prepare_tool_args(
        self, tool_name: str, raw_args: Dict, state: AgentState
    ) -> Tuple[str, Dict]:
        """Implementation for preparing tool call arguments."""
        return tool_name, raw_args

    def _process_tool_output(
        self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        """Process tool output and mutate state when appropriate."""
        if tool_call["name"] == "generate_test_code":
            test_code: Optional[str] = None
            qualified_class_name: Optional[str] = None

            test_code = result.get("test_code")
            qualified_class_name = result.get("qualified_class_name")

            if not isinstance(test_code, str) or not isinstance(
                qualified_class_name, str
            ):
                outputs.append(
                    ToolMessage(
                        content="Invalid output from generate_test_code tool.",
                        tool_call_id=tool_call["id"],
                    )
                )
                return

            project_root = self.project_root

            # Always delete the previously saved test file (if any) before saving the new one
            if (
                state.package is not None
                and state.class_name is not None
                and project_root is not None
            ):
                old_qcn = (
                    f"{state.package}.{state.class_name}"
                    if state.package
                    else state.class_name
                )
                old_info = TestFileInfo(qualified_class_name=old_qcn)

                # Attempt to delete the previous file
                fm = TestFileManager(project_root)
                deleted = fm.delete_single(old_info, encode_class_name=False)
                if not deleted:
                    expected_path = fm.target_path(old_info, encode_class_name=False)
                    raise FileDeletionError(
                        f"Failed to delete old test file at {expected_path}",
                        extra_info={
                            "qualified_class_name": old_qcn,
                            "path": str(expected_path),
                        },
                    )

            # Save new test code at the new location
            new_info = TestFileInfo(
                qualified_class_name=qualified_class_name,
                test_code=test_code,
            )
            saved_qcn, saved_path = TestFileManager(project_root).save_single(
                new_info, encode_class_name=False
            )
            outputs.append(
                ToolMessage(
                    content=f"Saved test code to {saved_qcn} at {saved_path}",
                    tool_call_id=tool_call["id"],
                )
            )

            # Update state with the saved package and simple class name
            if "." in saved_qcn:
                pkg, simple_cls = saved_qcn.rsplit(".", 1)
                state.package = pkg or None
                state.class_name = simple_cls
            else:
                state.package = None
                state.class_name = saved_qcn
            return

        if tool_call["name"] == "view_test_code":
            if not state.class_name:
                outputs.append(
                    ToolMessage(
                        content="No test code has been generated or saved.",
                        tool_call_id=tool_call["id"],
                    )
                )
                return

            qcn = (
                f"{state.package}.{state.class_name}"
                if state.package
                else state.class_name
            )
            fm = TestFileManager(self.project_root)
            info = TestFileInfo(qualified_class_name=qcn)
            try:
                raw_code = fm.load(info, encode_class_name=False)
                outputs.append(
                    ToolMessage(
                        content=raw_code,
                        tool_call_id=tool_call["id"],
                    )
                )
            except FileNotFoundError:
                outputs.append(
                    ToolMessage(
                        content=f"Test file not found for {qcn}.",
                        tool_call_id=tool_call["id"],
                    )
                )
            return

        if tool_call["name"] == "compile_and_execute_tests":
            # Compile, then execute test using state.package/state.class_name (non-encoded)
            if not state.class_name:
                outputs.append(
                    ToolMessage(
                        content="No test code has been generated or saved.",
                        tool_call_id=tool_call["id"],
                    )
                )
                return

            # Gather compile errors with detailed parsing
            erroneous_classes, compilation_errors = (
                JavaCompilation.get_erroneous_classes_and_errors(self.project_root)
            )
            file_key = f"{state.class_name}.java"
            has_error = any(ec == file_key for ec in erroneous_classes)

            comp_errors_dicts: List[Dict[str, Any]] = [
                e.model_dump() for e in compilation_errors
            ]
            target_errors: List[Dict[str, Any]] = [
                e for e in comp_errors_dicts if e.get("file", "").endswith(file_key)
            ]

            error_counts_by_file: Dict[str, int] = {}
            for e in comp_errors_dicts:
                f = (e.get("file") or "").strip()
                if f:
                    error_counts_by_file[f] = error_counts_by_file.get(f, 0) + 1

            any_compilation_errors = len(erroneous_classes) > 0

            result: Dict[str, Any] = {
                "compilation": {
                    "target_class_file": file_key,
                    "has_errors_for_target": has_error,
                    "any_compilation_errors": any_compilation_errors,
                    "errors_for_target_class": target_errors,
                    "error_summary": {
                        "total_errors": len(comp_errors_dicts),
                        "files_with_errors": sorted(list(error_counts_by_file.keys())),
                        "error_counts_by_file": error_counts_by_file,
                    },
                }
            }

            if not has_error:
                test_fqn = (
                    f"{state.package}.{state.class_name}"
                    if state.package
                    else state.class_name
                )
                execution_feedback = JavaExecution.execute(
                    str(self.project_root), test_fqn
                )

                issues: List[Dict[str, Any]] = [
                    i.model_dump() for i in execution_feedback.issues
                ]
                if execution_feedback.passed:
                    result["execution"] = {
                        "executed": True,
                        "status": "tests_passed",
                        "message": "All tests in class passed.",
                        "num_tests_run": execution_feedback.tests_run,
                        "num_failures": execution_feedback.failures,
                        "num_errors": execution_feedback.errors,
                    }
                else:
                    status = execution_feedback.failure_reason_code or (
                        "test_has_failures_or_errors"
                        if (execution_feedback.failures or execution_feedback.errors)
                        else "execution_failed"
                    )
                    reason = (
                        execution_feedback.failure_reason or "Test execution failed"
                    )
                    top_issues: List[str] = []
                    for it in issues[:10]:
                        name = (
                            f"{it.get('class_name','')}.{it.get('test_name','')}".strip(
                                "."
                            )
                        )
                        kind = it.get("kind")
                        msg = it.get("message") or it.get("error_type") or ""
                        loc = (
                            f" @ {it.get('file')}:{it.get('line')}"
                            if it.get("file") and it.get("line")
                            else ""
                        )
                        top_issues.append(f"{kind}: {name}{loc} -> {msg}")
                    result["execution"] = {
                        "executed": True,
                        "status": status,
                        "message": reason,
                        "num_tests_run": execution_feedback.tests_run,
                        "num_failures": execution_feedback.failures,
                        "num_errors": execution_feedback.errors,
                        "issues": top_issues,
                    }
            else:
                result["execution"] = {
                    "executed": False,
                    "status": "compilation_errors",
                    "message": "Fix compilation errors in target test class before execution.",
                }

            outputs.append(
                ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        if tool_call["name"] == "finalize":
            comments = result
            state.final_comments = str(comments)
            outputs.append(
                ToolMessage(content=str(comments), tool_call_id=tool_call["id"])
            )
            setattr(self, "_end_now", True)
            return

        if tool_call["name"] == "modify_scenario_comment":
            # Tool returns args directly
            step_id: Optional[int] = None
            comment: Optional[str] = None
            order: Optional[int] = None
            note: Optional[str] = None

            if isinstance(result, (list, tuple)) and len(result) == 2:
                # Disambiguate by which state is present
                if state.atomic_blocks is not None:
                    order, note = int(result[0]), str(result[1])
                else:
                    step_id, comment = int(result[0]), str(result[1])
            elif isinstance(result, dict):
                step_id = result.get("id")
                comment = result.get("comment")
                order = result.get("order")
                note = result.get("note")

            # GRAMMATICAL mode (flattened atomic blocks by order)
            if state.atomic_blocks is not None:
                if order is None or note is None:
                    outputs.append(
                        ToolMessage(
                            content="Invalid modify_scenario_comment args for GRAMMATICAL mode.",
                            tool_call_id=tool_call["id"],
                        )
                    )
                    return

                found = False
                for b in state.atomic_blocks.atomic_blocks:
                    if b.order == int(order):
                        b.notes = str(note)
                        found = True
                        break

                if found:
                    outputs.append(
                        ToolMessage(
                            content=f"Updated note for block order {order}.",
                            tool_call_id=tool_call["id"],
                        )
                    )
                else:
                    outputs.append(
                        ToolMessage(
                            content=f"No atomic block with order {order} exists.",
                            tool_call_id=tool_call["id"],
                        )
                    )
                return

            # GHERKIN mode (LocalizedScenario with LocalizedSteps)
            if state.localized_scenario is not None:
                if step_id is None or comment is None:
                    outputs.append(
                        ToolMessage(
                            content="Invalid modify_scenario_comment args for GHERKIN mode.",
                            tool_call_id=tool_call["id"],
                        )
                    )
                    return

                found = False
                # Search setup
                for s in state.localized_scenario.setup:
                    if s.id == int(step_id):
                        s.comments = str(comment)
                        found = True
                        break

                # Search steps: given/when/then
                if not found:
                    for gstep in state.localized_scenario.steps:
                        for group in (gstep.given, gstep.when, gstep.then):
                            for s in group:
                                if s.id == int(step_id):
                                    s.comments = str(comment)
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break

                # Search teardown
                if not found:
                    for s in state.localized_scenario.teardown:
                        if s.id == int(step_id):
                            s.comments = str(comment)
                            found = True
                            break

                if found:
                    outputs.append(
                        ToolMessage(
                            content=f"Updated comment for step id {step_id}.",
                            tool_call_id=tool_call["id"],
                        )
                    )
                else:
                    outputs.append(
                        ToolMessage(
                            content=f"No step with id {step_id} exists.",
                            tool_call_id=tool_call["id"],
                        )
                    )
                return

            # No recognizable state
            outputs.append(
                ToolMessage(
                    content="No scenario or atomic blocks present to modify.",
                    tool_call_id=tool_call["id"],
                )
            )
            return

        # Default: echo tool result
        outputs.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )
