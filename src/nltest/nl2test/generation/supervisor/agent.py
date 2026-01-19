from __future__ import annotations

from typing import List, Any, Dict, Tuple, Optional, Type
from pathlib import Path

from langchain_core.messages import ToolMessage, ToolCall
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from nltest.nl2test.core.react_agent import ReActAgent
from nltest.nl2test.core.message_redactor import MessageRedactor
from nltest.nl2test.generation.common.compilation_execution import (
    CompilationExecutionMixin,
)
from nltest.nl2test.generation.composition.orchestrators.base import (
    BaseCompositionOrchestrator,
)
from nltest.nl2test.generation.localization.orchestrators.base import (
    BaseLocalizationOrchestrator,
)
from nltest.nl2test.models import AgentState
from nltest.nl2test.models.agents import NoArgs
from nltest.utils.constants import TEST_DIR
from nltest.utils.llm import LLMClient
from nltest.utils.file_io.test_file_manager import TestFileManager, TestFileInfo
from nltest.utils.exceptions import ProjectCompilationError
from nltest.utils.tool_messages import format_tool_error, format_tool_ok


class SupervisorReActAgent(ReActAgent, CompilationExecutionMixin):
    """
    Supervisor agent that orchestrates localization and composition sub-agents.

    Coordinates the end-to-end conversion of natural language test descriptions
    into executable Java tests by delegating to specialized sub-agents.
    """

    # Redaction placeholders for token reduction
    _BLOCK_REDACTED = "(redacted due to updated blocks)"
    _SCENARIO_REDACTED = "(redacted due to updated scenario)"
    _CODE_REDACTED = "(redacted since new code generated)"

    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: List[BaseTool],
        system_message: str,
        nl_description: str,
        project_root: Path | str | None = None,
        test_base_dir: str | Path | None = None,
        module_root: Path | None = None,
        allow_duplicate_tools: List[BaseTool] | None = None,
        max_iters: int = 10,
        localization_agent: BaseLocalizationOrchestrator | None = None,
        composition_agent: BaseCompositionOrchestrator | None = None,
        parallelizable: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            llm=llm,
            tools=tools,
            allow_duplicate_tools=allow_duplicate_tools,
            system_message=system_message,
            allow_parallelize=parallelizable,
            max_iters=max_iters,
            **kwargs,
        )
        self._nl_description = nl_description
        self.project_root = Path(project_root) if project_root is not None else None
        self.test_base_dir = test_base_dir
        resolved_module_root = (
            Path(module_root).expanduser() if module_root is not None else None
        )
        if (
            resolved_module_root is not None
            and not resolved_module_root.is_absolute()
            and self.project_root is not None
        ):
            resolved_module_root = self.project_root / resolved_module_root
        self.module_root = (
            resolved_module_root.resolve() if resolved_module_root is not None else None
        )
        self.localization_agent = localization_agent
        self.composition_agent = composition_agent
        # Track last known states for reuse between calls
        self.localization_state: Optional[AgentState] = None
        self.composition_state: Optional[AgentState] = None

    def _get_finalize_schema(self) -> Type[BaseModel]:
        return NoArgs

    def _execute_force_end(self, state: AgentState) -> AgentState:
        """Skip LLM call for supervisor since NoArgs schema provides no value."""
        state.force_end_attempts += 1
        state.finalize_called = True
        state.final_comments = ""
        self._log_force_finalize(state)
        return state

    def _get_force_finalize_system_prompt(self) -> str:
        raise NotImplementedError(
            "Supervisor uses _execute_force_end override; prompt methods are unused."
        )

    def _get_force_finalize_chat_prompt(self) -> str:
        raise NotImplementedError(
            "Supervisor uses _execute_force_end override; prompt methods are unused."
        )

    def _process_force_finalize_result(
        self, result: BaseModel, state: AgentState
    ) -> None:
        raise NotImplementedError(
            "Supervisor uses _execute_force_end override; this method is unused."
        )

    def prepare_tool_args(
        self, tool_name: str, raw_args: Dict[str, Any], state: AgentState
    ) -> Tuple[str, Dict[str, Any]]:
        # No CLDK normalization needed - supervisor doesn't use static analysis tools
        return tool_name, raw_args

    def _clean_agent(
        self,
        state: Optional[AgentState],
        orchestrator: BaseLocalizationOrchestrator | BaseCompositionOrchestrator,
    ) -> Optional[AgentState]:
        """Reset sub-agent state for fresh invocation."""
        orchestrator.reset_agent()
        if state is None:
            return None
        cleaned = state.model_copy(deep=True)
        cleaned.reset_message_history()
        # TODO: Extend message cleaning to maybe use summaries of past history
        return cleaned

    def _get_test_file_manager(self) -> TestFileManager:
        test_base_dir = self.test_base_dir or TEST_DIR
        project_root = self.project_root or Path(".")
        return TestFileManager(project_root, test_base_dir=test_base_dir)

    def process_tool_output(
        self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        if self._append_tool_error_if_needed(tool_call, result, outputs):
            return

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

    def _redact_previous_block_outputs(self, state: AgentState) -> None:
        """Redact stale block/scenario payloads when new results are saved."""
        MessageRedactor.redact_tool_outputs(
            state,
            tool_names={"call_localization_agent", "call_composition_agent"},
            keys_to_redact={
                "blocks": self._BLOCK_REDACTED,
                "scenario": self._SCENARIO_REDACTED,
            },
        )

    def _redact_previous_test_code_results(self, state: AgentState) -> None:
        """Redact outdated tool outputs when new composition results arrive."""
        MessageRedactor.redact_tool_outputs_by_tool(
            state,
            tool_redactions={
                "view_test_code": {"source": self._CODE_REDACTED},
                "compile_and_execute_test": {
                    "compilation": self._CODE_REDACTED,
                    "execution": self._CODE_REDACTED,
                },
                "call_composition_agent": {
                    "package": self._CODE_REDACTED,
                    "class_name": self._CODE_REDACTED,
                    "method_signature": self._CODE_REDACTED,
                },
            },
        )

    def _process_call_localization_agent_output(
        self,
        tool_call: ToolCall,
        result: Dict[str, Any],
        state: AgentState,
        outputs: List,
    ) -> None:
        tool_name = tool_call["name"]
        instructions = result.get("instructions") or ""

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

        canonical_blocks = state.localized_scenario or state.atomic_blocks
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

        payload: Dict[str, Any] = {"comments": str(updated_state.final_comments or "")}
        if updated_state.atomic_blocks is not None:
            payload["blocks"] = updated_state.atomic_blocks.model_dump()
        elif updated_state.localized_scenario is not None:
            payload["scenario"] = updated_state.localized_scenario.model_dump()

        self._redact_previous_block_outputs(state)

        outputs.append(
            ToolMessage(
                content=format_tool_ok(payload),
                tool_call_id=tool_call["id"],
            )
        )

    def _process_call_composition_agent_output(
        self,
        tool_call: ToolCall,
        result: Dict[str, Any],
        state: AgentState,
        outputs: List,
    ) -> None:
        tool_name = tool_call["name"]
        instructions = result.get("instructions") or ""

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

        canonical_blocks = state.localized_scenario or state.atomic_blocks
        if canonical_blocks is None:
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code="missing_blocks",
                        message=(
                            "Supervisor has no scenario to inject into "
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
            payload["scenario"] = updated_state.localized_scenario.model_dump()

        self._redact_previous_block_outputs(state)
        self._redact_previous_test_code_results(state)

        outputs.append(
            ToolMessage(
                content=format_tool_ok(payload),
                tool_call_id=tool_call["id"],
            )
        )

    def _process_view_test_code_output(
        self,
        tool_call: ToolCall,
        result: Dict[str, Any],
        state: AgentState,
        outputs: List,
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

        if not isinstance(start_line, int) or not isinstance(end_line, int):
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code="invalid_line_range",
                        message="start_line and end_line must be integers.",
                        details={
                            "tool": tool_name,
                            "tool_call_id": tool_call["id"],
                            "start_line": start_line,
                            "end_line": end_line,
                        },
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        qcn = (
            f"{state.package}.{state.class_name}" if state.package else state.class_name
        )
        fm = self._get_test_file_manager()
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
        sliced_source = "\n".join(code_lines[start_line - 1 : applied_end_line])
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

    def _process_compile_and_execute_test_output(
        self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        """Process compile_and_execute_test using shared mixin."""
        self.process_compile_and_execute(tool_call, state, outputs)

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
        self._append_tool_output(tool_call, result, outputs)
