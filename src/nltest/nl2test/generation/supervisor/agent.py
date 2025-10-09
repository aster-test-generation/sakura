from __future__ import annotations

from typing import List, Any, Dict, Tuple, Optional
import json
from pathlib import Path

from langchain_core.messages import ToolMessage, ToolCall
from langchain_core.tools import BaseTool

from nltest.nl2test.core.react_agent import ReActAgent
from nltest.nl2test.generation.composition.orchestrators.base import (
    BaseCompositionOrchestrator,
)
from nltest.nl2test.generation.localization.orchestrators.base import (
    BaseLocalizationOrchestrator,
)
from nltest.nl2test.models import AgentState
from nltest.nl2test.models import LocalizedScenario, AtomicBlockList
from nltest.utils.llm import LLMClient
from nltest.utils.file_io.test_file_manager import TestFileManager, TestFileInfo
from nltest.utils.execution import JavaCompilation
from nltest.utils.execution.execution import JavaExecution


class SupervisorReActAgent(ReActAgent):
    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: List[BaseTool],
        system_message: str,
        nl_description: str,
        project_root: Path | str | None = None,
        allow_duplicate_tools: List[BaseTool] | None = None,
        max_iters: int = 10,
        localization_agent: BaseLocalizationOrchestrator | None = None,
        composition_agent: BaseCompositionOrchestrator | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            llm=llm,
            tools=tools,
            allow_duplicate_tools=allow_duplicate_tools,
            system_message=system_message,
            allow_parallelize=False,
            max_iters=max_iters,
            **kwargs,
        )
        self._nl_description = nl_description
        self.project_root = Path(project_root) if project_root is not None else None
        self.localization_agent = localization_agent
        self.composition_agent = composition_agent
        # Track last known states for reuse between calls
        self.localization_state: Optional[AgentState] = None
        self.composition_state: Optional[AgentState] = None

    def _clean_agent(
        self,
        state: Optional[AgentState],
        orchestrator: BaseLocalizationOrchestrator | BaseCompositionOrchestrator,
    ) -> Optional[AgentState]:
        orchestrator.reset_agent()
        if state is None:
            return None
        cleaned = state.model_copy(deep=True) if hasattr(state, "model_copy") else state
        cleaned.reset_message_history()
        # TODO: Extend message cleaning to maybe use summaries of past history
        return cleaned

    def _prepare_tool_args(
        self, tool_name: str, raw_args: Dict, state: AgentState
    ) -> Tuple[str, Dict]:
        return tool_name, raw_args

    def _process_tool_output(
        self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        name = tool_call["name"]

        # Delegate calls: Supervisor triggers underlying agents and returns normalized payload
        if name == "call_localization_agent":
            # Be robust to non-dict tool results
            if isinstance(result, dict):
                incoming_blocks = result.get("blocks")
                instructions = result.get("instructions")
            else:
                outputs.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "status": "error",
                                "reason": "unexpected_result_type",
                                "received_type": type(result).__name__,
                            }
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )
                return

            orchestrator = self.localization_agent
            if orchestrator is None:
                outputs.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "status": "error",
                                "reason": "localization_agent_not_configured",
                            }
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )
                return

            prev_state = self._clean_agent(self.localization_state, orchestrator)
            try:
                # Choose canonical blocks from state; ignore model-provided blocks except for first bootstrap
                canonical_blocks = None
                if self.localization_state and self.localization_state.localized_scenario is not None:
                    canonical_blocks = self.localization_state.localized_scenario
                elif state.localized_scenario is not None:
                    canonical_blocks = state.localized_scenario
                elif getattr(state, "atomic_blocks", None) is not None:
                    canonical_blocks = state.atomic_blocks
                else:
                    canonical_blocks = incoming_blocks

                updated_state: AgentState = orchestrator.assign_task(
                    canonical_blocks, instructions=instructions, agent_state=prev_state
                )
            except Exception as e:
                outputs.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "status": "error",
                                "reason": f"localization_agent_failed: {e}",
                            }
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )
                return

            # Persist latest localization state
            self.localization_state = updated_state

            # Update Supervisor state with blocks
            if updated_state.atomic_blocks is not None:
                state.atomic_blocks = updated_state.atomic_blocks
            if updated_state.localized_scenario is not None:
                state.localized_scenario = updated_state.localized_scenario

            payload: Dict[str, Any] = {
                "comments": str(updated_state.final_comments or "")
            }
            if updated_state.atomic_blocks is not None:
                payload["blocks"] = updated_state.atomic_blocks.model_dump()
            elif updated_state.localized_scenario is not None:
                payload["blocks"] = updated_state.localized_scenario.model_dump()

            outputs.append(
                ToolMessage(
                    content=json.dumps(payload),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        if name == "call_composition_agent":
            # Be robust to non-dict tool results
            if isinstance(result, dict):
                incoming_blocks = result.get("blocks")
                instructions = result.get("instructions")
            else:
                outputs.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "status": "error",
                                "reason": "unexpected_result_type",
                                "received_type": type(result).__name__,
                            }
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )
                return

            orchestrator = self.composition_agent
            if orchestrator is None:
                outputs.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "status": "error",
                                "reason": "composition_agent_not_configured",
                            }
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )
                return

            prev_state = self._clean_agent(self.composition_state, orchestrator)
            try:
                # Choose canonical blocks from state; ignore model-provided blocks except for first bootstrap
                canonical_blocks = None
                if self.composition_state and self.composition_state.localized_scenario is not None:
                    canonical_blocks = self.composition_state.localized_scenario
                elif state.localized_scenario is not None:
                    canonical_blocks = state.localized_scenario
                elif getattr(state, "atomic_blocks", None) is not None:
                    canonical_blocks = state.atomic_blocks
                else:
                    canonical_blocks = incoming_blocks

                updated_state: AgentState = orchestrator.assign_task(
                    canonical_blocks, instructions=instructions, agent_state=prev_state
                )
            except Exception as e:
                outputs.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "status": "error",
                                "reason": f"composition_agent_failed: {e}",
                            }
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )
                return

            # Persist latest composition state
            self.composition_state = updated_state

            # Update Supervisor state with blocks and packaging
            if updated_state.atomic_blocks is not None:
                state.atomic_blocks = updated_state.atomic_blocks
            if updated_state.localized_scenario is not None:
                state.localized_scenario = updated_state.localized_scenario
            state.package = updated_state.package
            state.class_name = updated_state.class_name
            state.method_signature = updated_state.method_signature

            payload: Dict[str, Any] = {
                "comments": str(updated_state.final_comments or ""),
                "package": updated_state.package,
                "class_name": updated_state.class_name,
                "method_signature": updated_state.method_signature,
            }
            if updated_state.atomic_blocks is not None:
                payload["blocks"] = updated_state.atomic_blocks.model_dump()
            elif updated_state.localized_scenario is not None:
                payload["blocks"] = updated_state.localized_scenario.model_dump()

            outputs.append(
                ToolMessage(
                    content=json.dumps(payload),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        if name == "view_test_code":
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
            fm = TestFileManager(self.project_root or Path("."))
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

        if name == "compile_and_execute_test":
            if not state.class_name:
                outputs.append(
                    ToolMessage(
                        content="No test code has been generated or saved.",
                        tool_call_id=tool_call["id"],
                    )
                )
                return

            project_root = self.project_root or Path(".")
            erroneous_files, compilation_errors = (
                JavaCompilation.get_erroneous_files_and_errors(project_root)
            )
            file_key = f"{state.class_name}.java"
            has_error = any(ef == file_key for ef in erroneous_files)

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

            any_compilation_errors = len(erroneous_files) > 0

            result_payload: Dict[str, Any] = {
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
                execution_feedback = JavaExecution.execute(str(project_root), test_fqn)

                issues: List[Dict[str, Any]] = [
                    i.model_dump() for i in execution_feedback.issues
                ]
                if execution_feedback.passed:
                    result_payload["execution"] = {
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
                    result_payload["execution"] = {
                        "executed": True,
                        "status": status,
                        "message": reason,
                        "num_tests_run": execution_feedback.tests_run,
                        "num_failures": execution_feedback.failures,
                        "num_errors": execution_feedback.errors,
                        "issues": top_issues,
                    }
            else:
                result_payload["execution"] = {
                    "executed": False,
                    "status": "compilation_errors",
                    "message": "Fix compilation errors in target test class before execution.",
                }

            outputs.append(
                ToolMessage(
                    content=json.dumps(result_payload),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        if name == "finalize":
            # Mark conclusion of the agent; capture final comments if provided
            state.final_comments = (
                str(result) if result is not None else (state.final_comments or "")
            )
            outputs.append(
                ToolMessage(
                    content=str(result or "finalized"), tool_call_id=tool_call["id"]
                )
            )
            setattr(self, "_end_now", True)
            return

        # Default passthrough
        outputs.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )
