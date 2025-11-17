from __future__ import annotations

from typing import List, Any, Dict, Tuple, Optional
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
from nltest.utils.analysis.common_analysis import CommonAnalysis
from nltest.utils.compilation.maven import CompilationError, JavaMavenCompilation
from nltest.utils.execution.maven import ExecutionIssue, JavaMavenExecution
from nltest.utils.llm import LLMClient
from nltest.utils.file_io.test_file_manager import TestFileManager, TestFileInfo
from nltest.utils.exceptions import ProjectCompilationError
from nltest.utils.tool_messages import format_tool_error, format_tool_ok


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

    def prepare_tool_args(
            self, tool_name: str, raw_args: Dict[str, Any], _state: AgentState
    ) -> Tuple[str, Dict[str, Any]]:
        # Note: No need to normalize method sig to account for CLDK constructor calls since no static analysis tools
        return tool_name, raw_args

    def _clean_agent(
            self,
            state: Optional[AgentState],
            orchestrator: BaseLocalizationOrchestrator | BaseCompositionOrchestrator,
    ) -> Optional[AgentState]:
        orchestrator.reset_agent()
        if state is None:
            return None
        cleaned = state.model_copy(deep=True)
        cleaned.reset_message_history()
        # TODO: Extend message cleaning to maybe use summaries of past history
        return cleaned

    def process_tool_output(
            self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        tool_name = tool_call["name"]
        handler_map = {
            "call_localization_agent": self._process_call_localization_agent_output,
            "call_composition_agent": self._process_call_composition_agent_output,
            "view_test_code": self._process_view_test_code_output,
            "compile_and_execute_test": self._process_compile_and_execute_test_output,
            "finalize": self._process_finalize_tool_output,
        }
        handler = handler_map.get(tool_name, self.process_generic_tool_output)

        try:
            handler(tool_call, result, state, outputs)
        except ProjectCompilationError:
            raise
        except Exception as exc:
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

    def _process_call_localization_agent_output(
            self, tool_call: ToolCall, result: Dict[str, Any], state: AgentState, outputs: List
    ) -> None:
        tool_name = tool_call["name"]
        instructions = result.get("instructions")

        orchestrator = self.localization_agent
        if orchestrator is None:
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code="localization_agent_not_configured",
                        message="Localization agent is not configured on the supervisor.",
                        details={
                            "tool": tool_name,
                            "tool_call_id": tool_call["id"],
                        },
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        prev_state = self._clean_agent(self.localization_state, orchestrator)

        canonical_blocks = None
        if state.localized_scenario is not None:
            canonical_blocks = state.localized_scenario
        elif state.atomic_blocks is not None:
            canonical_blocks = state.atomic_blocks

        if canonical_blocks is None:
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code="missing_blocks",
                        message=(
                            "Supervisor has no blocks to inject into "
                            "call_localization_agent. Ensure the supervisor state is initialized."
                        ),
                        details={
                            "tool": tool_name,
                            "tool_call_id": tool_call["id"],
                        },
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
            return
        try:
            updated_state: AgentState = orchestrator.assign_task(
                canonical_blocks, instructions=instructions, agent_state=prev_state
            )
        except Exception as exc:
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code="localization_agent_failed",
                        message=str(exc),
                        details={
                            "tool": tool_name,
                            "tool_call_id": tool_call["id"],
                        },
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        if updated_state.curr_tool_trajectory:
            updated_state.tool_trajectories.append(
                updated_state.curr_tool_trajectory.copy()
            )
            updated_state.curr_tool_trajectory.clear()

        self.localization_state = updated_state

        if updated_state.atomic_blocks is not None:
            if hasattr(updated_state.atomic_blocks, "enforce_candidate_limits"):
                updated_state.atomic_blocks.enforce_candidate_limits()
            state.atomic_blocks = updated_state.atomic_blocks
        if updated_state.localized_scenario is not None:
            if hasattr(updated_state.localized_scenario, "enforce_candidate_limits"):
                updated_state.localized_scenario.enforce_candidate_limits()
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
                content=format_tool_ok(payload),
                tool_call_id=tool_call["id"],
            )
        )

    def _process_call_composition_agent_output(
            self, tool_call: ToolCall, result: Dict[str, Any], state: AgentState, outputs: List
    ) -> None:
        tool_name = tool_call["name"]
        instructions = result.get("instructions")

        orchestrator = self.composition_agent
        if orchestrator is None:
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code="composition_agent_not_configured",
                        message="Composition agent is not configured on the supervisor.",
                        details={
                            "tool": tool_name,
                            "tool_call_id": tool_call["id"],
                        },
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        prev_state = self._clean_agent(self.composition_state, orchestrator)

        canonical_blocks = None
        if state.localized_scenario is not None:
            canonical_blocks = state.localized_scenario
        elif state.atomic_blocks is not None:
            canonical_blocks = state.atomic_blocks

        if canonical_blocks is None:
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code="missing_blocks",
                        message=(
                            "Supervisor has no blocks to inject into "
                            "call_composition_agent. Ensure the supervisor state is initialized."
                        ),
                        details={
                            "tool": tool_name,
                            "tool_call_id": tool_call["id"],
                        },
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
            return
        try:
            updated_state: AgentState = orchestrator.assign_task(
                canonical_blocks, instructions=instructions, agent_state=prev_state
            )
        except Exception as exc:
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code="composition_agent_failed",
                        message=str(exc),
                        details={
                            "tool": tool_name,
                            "tool_call_id": tool_call["id"],
                        },
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        if updated_state.curr_tool_trajectory:
            updated_state.tool_trajectories.append(
                updated_state.curr_tool_trajectory.copy()
            )
            updated_state.curr_tool_trajectory.clear()

        self.composition_state = updated_state

        if updated_state.atomic_blocks is not None:
            updated_state.atomic_blocks.enforce_candidate_limits()
            state.atomic_blocks = updated_state.atomic_blocks
        if updated_state.localized_scenario is not None:
            updated_state.localized_scenario.enforce_candidate_limits()
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
                content=format_tool_ok(payload),
                tool_call_id=tool_call["id"],
            )
        )

    def _process_view_test_code_output(
            self, tool_call: ToolCall, result: Dict[str, Any], state: AgentState, outputs: List
    ) -> None:
        tool_name = tool_call["name"]
        if not state.class_name:
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code="no_active_test",
                        message="No test code has been generated or saved.",
                        details={
                            "tool": tool_name,
                            "tool_call_id": tool_call["id"],
                        },
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        start_line = result.get("start_line")
        end_line = result.get("end_line")

        qcn = (
            f"{state.package}.{state.class_name}"
            if state.package
            else state.class_name
        )
        fm = TestFileManager(self.project_root or Path("."))
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

    def _format_compilation_error(self, compilation_error: CompilationError) -> str:
        line = compilation_error.line if compilation_error.line is not None else "Unknown"

        if compilation_error.details:
            details_text = "\n".join(compilation_error.details)
        else:
            details_text = "No compiler details were provided."

        return (
            f"Line: {line}\n"
            f"Error Message: {compilation_error.message}\n"
            f"Error Details:\n{details_text}"
        )

    def _format_execution_issue(self, execution_issue: ExecutionIssue) -> str:
        class_name = execution_issue.class_name
        test_name = execution_issue.test_name
        test_case = f"{class_name}.{test_name}" if class_name and test_name else None

        line = execution_issue.line if execution_issue.line is not None else "Unknown"

        issue_type = execution_issue.kind or "Unknown"
        message = execution_issue.message or "No execution message was provided."
        stack_trace = execution_issue.stack_trace.strip() if execution_issue.stack_trace else "No stack trace was captured."

        return (
            f"Test Case: {test_case}\n"
            f"Issue Kind: {issue_type}\n"
            f"Message: {message}\n"
            f"Line: {line}\n"
            f"Stack Trace:\n{stack_trace}"
        )

    def _process_compile_and_execute_test_output(
            self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        tool_name = tool_call["name"]
        if not state.class_name or not state.method_signature:
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

        compilation_errors: List[CompilationError] = JavaMavenCompilation(self.project_root).get_compilation_errors()
        file_key = f"{state.class_name}.java"
        has_error_for_project = len(compilation_errors) > 0
        has_error_for_target = any(ef.file.endswith(file_key) for ef in compilation_errors)

        target_errors: List[str] = [self._format_compilation_error(compilation_error) for compilation_error in
                                    compilation_errors if compilation_error.file.endswith(file_key)]

        result_payload: Dict[str, Any] = {
            "compilation": {
                "status": "success" if not has_error_for_project else "compilation_error",
                "target_class_file": file_key,
                "has_errors_for_project": has_error_for_project,
                "has_errors_for_target": has_error_for_target,
                "error_details_for_target_class": target_errors,
            }
        }

        if has_error_for_project and not has_error_for_target:
            files_with_errors = [compilation_error.file for compilation_error in compilation_errors]
            error_details = [self._format_compilation_error(compilation_error) for compilation_error in
                             compilation_errors]
            raise ProjectCompilationError(
                "Unrelated project files failed to compile.",
                extra_info={
                    "files_with_errors": sorted(files_with_errors),
                    "error_details": error_details
                },
            )

        if not has_error_for_target:
            qualified_class_name = f"{state.package}.{state.class_name}" if state.package else state.class_name
            method_signature = state.method_signature
            execution_issues: List[ExecutionIssue] = JavaMavenExecution(self.project_root).get_execution_errors(
                qualified_class_name,
                method_signature)
            has_exec_error = len(execution_issues) > 0
            execution_failures: List[ExecutionIssue] = [execution_issue for execution_issue in execution_issues if
                                                        execution_issue.kind == "failure"]
            execution_errors: List[ExecutionIssue] = [execution_issue for execution_issue in execution_issues if
                                                      execution_issue.kind == "error"]

            result_payload["execution"] = {
                "status": "success" if not has_exec_error else "execution_error",
                "num_failures": len(execution_failures),
                "num_errors": len(execution_errors),
                "execution_failures": [self._format_execution_issue(execution_issue) for execution_issue in
                                       execution_failures],
                "execution_errors": [self._format_execution_issue(execution_issue) for execution_issue in
                                     execution_errors],
            }

        else:
            result_payload["execution"] = {
                "status": "compilation_errors",
                "message": "Fix compilation errors in target test class before execution.",
            }

        outputs.append(
            ToolMessage(
                content=format_tool_ok(result_payload),
                tool_call_id=tool_call["id"],
            )
        )

    def _process_finalize_tool_output(
            self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        state.final_comments = (
            str(result) if result is not None else (state.final_comments or "")
        )
        state.finalize_called = True
        state.force_end_attempts = 0
        outputs.append(
            ToolMessage(
                content=format_tool_ok(
                    {"comments": str(state.final_comments or "finalized")}
                ),
                tool_call_id=tool_call["id"],
            )
        )
        setattr(self, "_end_now", True)

    def process_generic_tool_output(
            self, tool_call: ToolCall, result: Any, _: AgentState, outputs: List
    ) -> None:
        outputs.append(
            ToolMessage(
                content=format_tool_ok(result),
                tool_call_id=tool_call["id"],
            )
        )
