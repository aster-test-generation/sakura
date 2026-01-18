from __future__ import annotations

from typing import Any, Dict, List, Tuple, Type

from langchain_core.messages import ToolMessage, ToolCall
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from nltest.nl2test.core.react_agent import ReActAgent
from nltest.nl2test.generation.common.cldk_normalizer import CLDKArgNormalizer
from nltest.nl2test.models import AgentState
from nltest.nl2test.models.agents import FinalizeScenarioArgs, FinalizeAtomicBlockArgs
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.utils.llm import LLMClient
from nltest.utils.tool_messages import format_tool_ok, format_tool_error


class LocalizationReActAgent(ReActAgent):
    """
    Localization agent that maps natural language steps to relevant code methods.

    Decomposes and localizes test description steps to concrete methods in the
    target Java application using vector search and static analysis.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: List[BaseTool],
        allow_duplicate_tools: List[BaseTool] | None = None,
        system_message: str,
        max_iters: int = 30,
        decomposition_mode: DecompositionMode = DecompositionMode.GRAMMATICAL,
        parallelizable: bool = True,
    ):
        super().__init__(
            llm=llm,
            tools=tools,
            allow_duplicate_tools=allow_duplicate_tools,
            system_message=system_message,
            allow_parallelize=parallelizable,
            max_iters=max_iters,
        )
        self.decomposition_mode = decomposition_mode

    def _get_finalize_schema(self) -> Type[BaseModel]:
        if self.decomposition_mode == DecompositionMode.GHERKIN:
            return FinalizeScenarioArgs
        return FinalizeAtomicBlockArgs

    def _get_force_finalize_system_prompt(self) -> str:
        if self.decomposition_mode == DecompositionMode.GHERKIN:
            template = LoadPrompt.load_prompt(
                "localization_agent_gherkin_finalize.jinja2",
                PromptFormat.JINJA2,
                "system",
            )
            return template.format()
        return (
            "You must produce the finalize output for the localization task. "
            "Use the message history to extract the current state of the localized steps."
        )

    def _get_force_finalize_chat_prompt(self) -> str:
        if self.decomposition_mode == DecompositionMode.GHERKIN:
            template = LoadPrompt.load_prompt(
                "localization_agent_gherkin_finalize.jinja2",
                PromptFormat.JINJA2,
                "chat",
            )
            return template.format()
        return "Produce the final localized blocks based on the conversation history."

    def _process_force_finalize_result(
        self, result: BaseModel, state: AgentState
    ) -> None:
        state.finalize_called = True
        state.final_comments = getattr(result, "comments", "")

        if self.decomposition_mode == DecompositionMode.GHERKIN:
            if hasattr(result, "localized_scenario"):
                localized = getattr(result, "localized_scenario")
                localized.enforce_candidate_limits()
                state.localized_scenario = localized
        else:
            if hasattr(result, "current_blocks"):
                blocks = getattr(result, "current_blocks")
                blocks.enforce_candidate_limits()
                state.atomic_blocks = blocks

    def prepare_tool_args(
        self, tool_name: str, raw_args: Dict[str, Any], state: AgentState
    ) -> Tuple[str, Dict[str, Any]]:
        """Normalize tool arguments for CLDK compatibility."""
        _ = state
        updated_args = CLDKArgNormalizer.normalize_args(tool_name, raw_args)
        return tool_name, updated_args

    def process_tool_output(
        self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        if self._append_tool_error_if_needed(tool_call, result, outputs):
            return

        tool_name = tool_call["name"]
        handler = {
            "finalize": self._process_finalize_tool_output,
        }.get(tool_name, self._process_generic_tool_output)

        try:
            handler(tool_call, result, state, outputs)
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

    def _process_finalize_tool_output(
        self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        blocks, comments = result
        state.final_comments = str(comments)
        state.finalize_called = True
        state.force_end_attempts = 0

        if self.decomposition_mode == DecompositionMode.GHERKIN:
            blocks.enforce_candidate_limits()
            state.localized_scenario = blocks  # Expected LocalizedScenario
        else:
            blocks.enforce_candidate_limits()
            state.atomic_blocks = blocks  # Expected AtomicBlockList

        outputs.append(
            ToolMessage(
                content=format_tool_ok({"comments": str(comments)}),
                tool_call_id=tool_call["id"],
            )
        )
        setattr(self, "_end_now", True)

    def _process_generic_tool_output(
        self, tool_call: ToolCall, result: Any, _: AgentState, outputs: List
    ) -> None:
        self._append_tool_output(tool_call, result, outputs)
