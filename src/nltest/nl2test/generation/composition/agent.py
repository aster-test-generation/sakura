from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

from langchain_core.messages import ToolMessage, ToolCall
from langchain_core.tools import BaseTool

from nltest.nl2test.core.react_agent import ReActAgent
from nltest.nl2test.models import AgentState, DecompositionMode
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.utils.llm import LLMClient
from nltest.utils.file_io.test_file_manager import TestFileManager, TestFileInfo
from nltest.utils.execution import JavaCompilation
from nltest.utils.execution.execution import JavaExecution
from nltest.utils.exceptions import ProjectCompilationError
from nltest.utils.tool_messages import format_tool_error, format_tool_ok


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

    @staticmethod
    def _cleanup_empty_dirs(base_dir: Path, start_dir: Path) -> None:
        try:
            base = base_dir.resolve()
            current = start_dir.resolve()
        except Exception:
            return

        while current != base and base in current.parents:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    def _prepare_tool_args(
            self, tool_name: str, raw_args: Dict, state: AgentState
    ) -> Tuple[str, Dict]:
        """Implementation for preparing tool call arguments."""
        return tool_name, raw_args

    def _process_tool_output(
            self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        """Process tool output and mutate state when appropriate."""
        tool_name = tool_call["name"]
        try:
            if tool_name == "generate_test_code":
                test_code = result.get("test_code")
                qualified_class_name = result.get("qualified_class_name")
                method_signature = result.get("method_signature")

                fm = TestFileManager(self.project_root)

                old_info: Optional[TestFileInfo] = None
                old_path: Optional[Path] = None
                if state.package and state.class_name:
                    old_qcn = f"{state.package}.{state.class_name}" if state.package else state.class_name
                    old_info = TestFileInfo(qualified_class_name=old_qcn)
                    old_path = fm.target_path(old_info, encode_class_name=False)

                new_info = TestFileInfo(qualified_class_name=qualified_class_name, test_code=test_code)
                target_path = fm.target_path(new_info, encode_class_name=False)

                if old_path is not None and old_path == target_path:
                    saved_qcn, saved_path = fm.save_single(
                        new_info, encode_class_name=False, allow_overwrite=True
                    )
                else:
                    if old_info is not None and old_path is not None:
                        fm.delete_single(
                            old_info,
                            encode_class_name=False,
                            strict=True,
                            max_attempts=2,
                            retry_delay=0.05,
                        )
                        self._cleanup_empty_dirs(fm.test_base_dir, old_path.parent)

                    saved_qcn, saved_path = fm.save_single(new_info, encode_class_name=False)

                outputs.append(
                    ToolMessage(
                        content=format_tool_ok(
                            {
                                "message": "Saved test code.",
                                "qualified_class_name": saved_qcn,
                                "path": str(saved_path),
                            }
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )

                if "." in saved_qcn:
                    pkg, simple_cls = saved_qcn.rsplit(".", 1)
                    state.package = pkg or None
                    state.class_name = simple_cls
                else:
                    state.package = None
                    state.class_name = saved_qcn
                state.method_signature = method_signature.strip() if method_signature.strip() else None
                return

            elif tool_name == "view_test_code":
                if not state.class_name:
                    outputs.append(
                        ToolMessage(
                            content=format_tool_error(
                                code="no_active_test",
                                message="No test code has been generated or saved.",
                                details={"tool": tool_name, "tool_call_id": tool_call["id"]},
                            ),
                            tool_call_id=tool_call["id"],
                        )
                    )
                    return

                start_line = result.get("start_line")
                end_line = result.get("end_line")

                qcn = f"{state.package}.{state.class_name}" if state.package else state.class_name
                fm = TestFileManager(self.project_root)
                info = TestFileInfo(qualified_class_name=qcn)
                try:
                    raw_code = fm.load(info, encode_class_name=False)
                except FileNotFoundError:
                    outputs.append(
                        ToolMessage(
                            content=format_tool_error(
                                code="test_file_not_found",
                                message=f"Test file not found for {qcn}.",
                                details={
                                    "tool": tool_name,
                                    "tool_call_id": tool_call["id"],
                                    "qualified_class_name": qcn,
                                },
                            ),
                            tool_call_id=tool_call["id"],
                        )
                    )
                    return

                code_lines = raw_code.splitlines()
                total_lines = len(code_lines)
                if start_line > total_lines:
                    outputs.append(
                        ToolMessage(
                            content=format_tool_error(
                                code="invalid_line_range",
                                message="start_line is beyond the end of the file.",
                                details={
                                    "tool": tool_name,
                                    "tool_call_id": tool_call["id"],
                                    "start_line": start_line,
                                    "total_lines": total_lines,
                                },
                            ),
                            tool_call_id=tool_call["id"],
                        )
                    )
                    return

                applied_end_line = min(end_line, total_lines)
                sliced_source = "\n".join(
                    code_lines[start_line - 1:applied_end_line]
                )
                payload = {
                    "qualified_class_name": qcn,
                    "source": sliced_source,
                    "total_lines": total_lines,
                    "start_line": start_line,
                    "end_line": applied_end_line,
                }

                outputs.append(
                    ToolMessage(
                        content=format_tool_ok(payload),
                        tool_call_id=tool_call["id"],
                    )
                )
                return

            elif tool_name == "compile_and_execute_test":
                if not state.class_name:
                    outputs.append(
                        ToolMessage(
                            content=format_tool_error(
                                code="no_active_test",
                                message="No test code has been generated or saved.",
                                details={"tool": tool_name, "tool_call_id": tool_call["id"]},
                            ),
                            tool_call_id=tool_call["id"],
                        )
                    )
                    return

                erroneous_files, compilation_errors = (
                    JavaCompilation.get_erroneous_files_and_errors(self.project_root)
                )
                file_key = f"{state.class_name}.java"
                has_error = any(ef == file_key for ef in erroneous_files)

                comp_errors_dicts: List[Dict[str, Any]] = [entry.model_dump() for entry in compilation_errors]
                target_errors: List[Dict[str, Any]] = [
                    entry for entry in comp_errors_dicts if entry.get("file", "").endswith(file_key)
                ]

                error_counts_by_file: Dict[str, int] = {}
                for entry in comp_errors_dicts:
                    file_name = (entry.get("file") or "").strip()
                    if file_name:
                        error_counts_by_file[file_name] = error_counts_by_file.get(file_name, 0) + 1

                any_compilation_errors = bool(erroneous_files)

                result_payload: Dict[str, Any] = {
                    "compilation": {
                        "target_class_file": file_key,
                        "has_errors_for_target": has_error,
                        "any_compilation_errors": any_compilation_errors,
                        "errors_for_target_class": target_errors,
                        "error_summary": {
                            "total_errors": len(comp_errors_dicts),
                            "files_with_errors": sorted(error_counts_by_file.keys()),
                            "error_counts_by_file": error_counts_by_file,
                        },
                    }
                }

                if any_compilation_errors and not has_error:
                    raise ProjectCompilationError(
                        "Unrelated project files failed to compile.",
                        extra_info={
                            "files_with_errors": sorted(error_counts_by_file.keys()),
                            "error_counts_by_file": error_counts_by_file,
                        },
                    )

                if not has_error:
                    test_fqn = f"{state.package}.{state.class_name}" if state.package else state.class_name
                    execution_feedback = JavaExecution.execute(str(self.project_root), test_fqn)

                    issues: List[Dict[str, Any]] = [issue.model_dump() for issue in execution_feedback.issues]
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
                        reason = execution_feedback.failure_reason or "Test execution failed"
                        top_issues: List[str] = []
                        for issue in issues[:10]:
                            name = f"{issue.get('class_name', '')}.{issue.get('test_name', '')}".strip(".")
                            kind = issue.get("kind")
                            message_text = issue.get("message") or issue.get("error_type") or ""
                            location = (
                                f" @ {issue.get('file')}:{issue.get('line')}"
                                if issue.get("file") and issue.get("line")
                                else ""
                            )
                            top_issues.append(f"{kind}: {name}{location} -> {message_text}")
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
                        content=format_tool_ok(result_payload),
                        tool_call_id=tool_call["id"],
                    )
                )
                return

            elif tool_name == "finalize":
                comments = result
                state.final_comments = str(comments)
                state.finalize_called = True
                state.force_end_attempts = 0
                outputs.append(
                    ToolMessage(
                        content=format_tool_ok({"comments": str(comments)}),
                        tool_call_id=tool_call["id"],
                    )
                )
                setattr(self, "_end_now", True)
                return

            elif tool_name == "modify_scenario_comment":
                step_id: Optional[int] = None
                comment: Optional[str] = None
                order: Optional[int] = None
                note: Optional[str] = None

                if isinstance(result, (list, tuple)) and len(result) == 2:
                    if state.atomic_blocks is not None:
                        order, note = int(result[0]), str(result[1])
                    else:
                        step_id, comment = int(result[0]), str(result[1])
                elif isinstance(result, dict):
                    step_id = result.get("id")
                    comment = result.get("comment")
                    order = result.get("order")
                    note = result.get("note")

                if state.atomic_blocks is not None:
                    if order is None or note is None:
                        outputs.append(
                            ToolMessage(
                                content=format_tool_error(
                                    code="invalid_arguments",
                                    message="Invalid modify_scenario_comment args for GRAMMATICAL mode.",
                                    details={
                                        "tool": tool_name,
                                        "tool_call_id": tool_call["id"],
                                        "order": order,
                                    },
                                ),
                                tool_call_id=tool_call["id"],
                            )
                        )
                        return

                    found = False
                    for block in state.atomic_blocks.atomic_blocks:
                        if block.order == int(order):
                            block.notes = str(note)
                            found = True
                            break

                    if found:
                        outputs.append(
                            ToolMessage(
                                content=format_tool_ok(
                                    {"message": f"Updated note for block order {order}.", "order": order}
                                ),
                                tool_call_id=tool_call["id"],
                            )
                        )
                    else:
                        outputs.append(
                            ToolMessage(
                                content=format_tool_error(
                                    code="block_not_found",
                                    message=f"No atomic block with order {order} exists.",
                                    details={
                                        "tool": tool_name,
                                        "tool_call_id": tool_call["id"],
                                        "order": order,
                                    },
                                ),
                                tool_call_id=tool_call["id"],
                            )
                        )
                    return

                if state.localized_scenario is not None:
                    if step_id is None or comment is None:
                        outputs.append(
                            ToolMessage(
                                content=format_tool_error(
                                    code="invalid_arguments",
                                    message="Invalid modify_scenario_comment args for GHERKIN mode.",
                                    details={
                                        "tool": tool_name,
                                        "tool_call_id": tool_call["id"],
                                        "step_id": step_id,
                                    },
                                ),
                                tool_call_id=tool_call["id"],
                            )
                        )
                        return

                    found = False
                    for entry in state.localized_scenario.setup:
                        if entry.id == int(step_id):
                            entry.comments = str(comment)
                            found = True
                            break

                    if not found:
                        for grouped in state.localized_scenario.steps:
                            for cluster in (grouped.given, grouped.when, grouped.then):
                                for entry in cluster:
                                    if entry.id == int(step_id):
                                        entry.comments = str(comment)
                                        found = True
                                        break
                                if found:
                                    break
                            if found:
                                break

                    if not found:
                        for entry in state.localized_scenario.teardown:
                            if entry.id == int(step_id):
                                entry.comments = str(comment)
                                found = True
                                break

                    if found:
                        outputs.append(
                            ToolMessage(
                                content=format_tool_ok(
                                    {"message": f"Updated comment for step id {step_id}.", "step_id": step_id}
                                ),
                                tool_call_id=tool_call["id"],
                            )
                        )
                    else:
                        outputs.append(
                            ToolMessage(
                                content=format_tool_error(
                                    code="step_not_found",
                                    message=f"No step with id {step_id} exists.",
                                    details={
                                        "tool": tool_name,
                                        "tool_call_id": tool_call["id"],
                                        "step_id": step_id,
                                    },
                                ),
                                tool_call_id=tool_call["id"],
                            )
                        )
                    return

                outputs.append(
                    ToolMessage(
                        content=format_tool_error(
                            code="state_unavailable",
                            message="No scenario or atomic blocks present to modify.",
                            details={
                                "tool": tool_name,
                                "tool_call_id": tool_call["id"],
                            },
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )
                return

            outputs.append(
                ToolMessage(
                    content=format_tool_ok(result),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        except ProjectCompilationError:
            raise

        except Exception as exc:  # pragma: no cover - defensive
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code=type(exc).__name__,
                        message=str(exc),
                        details={
                            "tool": tool_name,
                            "tool_call_id": tool_call["id"],
                        },
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
