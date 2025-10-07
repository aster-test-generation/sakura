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
        # Stopped injecting from state to reduce tool complexity

        # if tool_name == "modify_atomic_blocks":
        #    raw_args = dict(raw_args)
        #    raw_args.setdefault("current_blocks", getattr(state, "atomic_blocks", AtomicBlockList(atomic_blocks=[])))
        # elif tool_name == "finalize":
        #    raw_args = dict(raw_args)
        #    # Support either Gherkin Scenario or AtomicBlockList depending on the flow
        #    if "scenario" not in raw_args and getattr(state, "scenario", None) is not None:
        #        raw_args.setdefault("scenario", state.scenario)
        #    else:
        #        raw_args.setdefault("current_blocks", getattr(state, "atomic_blocks", AtomicBlockList(atomic_blocks=[])))

        return tool_name, raw_args

    def _process_tool_output(
        self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        # Finalize and end based on decomposition mode; avoid per-type isinstance checks
        if tool_call["name"] == "finalize":
            blocks, comments = result
            state.final_comments = str(comments)

            if self.decomposition_mode == DecompositionMode.GHERKIN:
                state.localized_scenario = blocks  # Expected LocalizedScenario
            else:
                state.atomic_blocks = blocks  # Expected AtomicBlockList

            outputs.append(
                ToolMessage(content=str(comments), tool_call_id=tool_call["id"])
            )
            # End the agent
            setattr(self, "_end_now", True)
            return

        # Default: just surface the tool result
        outputs.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
