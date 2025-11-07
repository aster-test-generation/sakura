from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.messages import ToolMessage, ToolCall
from langchain_core.tools import BaseTool

from nltest.nl2test.core.react_agent import ReActAgent  # base in this repo
from nltest.nl2test.models import (
    AgentState,
    AtomicBlock,
    AtomicBlockList,
    Scenario,
    LocalizedScenario,
)
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.utils.llm import LLMClient
from nltest.utils.tool_messages import format_tool_ok, format_tool_error


class LocalizationReActAgent(ReActAgent):
    def __init__(
            self,
            *,
            llm: LLMClient,
            tools: List[BaseTool],
            allow_duplicate_tools: List[BaseTool] | None = None,
            system_message: str,
            max_iters: int = 30,
            decomposition_mode: DecompositionMode = DecompositionMode.GRAMMATICAL,
    ):
        super().__init__(
            llm=llm,
            tools=tools,
            allow_duplicate_tools=allow_duplicate_tools,
            system_message=system_message,
            allow_parallelize=True,
            max_iters=max_iters,
        )
        self.decomposition_mode = decomposition_mode

    def _prepare_tool_args(
            self, tool_name: str, raw_args: Dict, state: AgentState
    ) -> Tuple[str, Dict]:
        return tool_name, raw_args

    def _process_tool_output(
            self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        # Finalize and end based on decomposition mode; avoid per-type isinstance checks
        tool_name = tool_call["name"]
        try:
            if tool_name == "finalize":
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
                return

            outputs.append(
                ToolMessage(
                    content=format_tool_ok(result),
                    tool_call_id=tool_call["id"],
                )
            )
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
