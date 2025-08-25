from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.messages import ToolMessage, ToolCall
from langchain_core.tools import BaseTool

from nltest.nl2test.core.react_agent import ReActAgent  # base in this repo
from nltest.nl2test.model.models import AgentState, AtomicBlock
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.utils.llm.llm_client import LLMClient


class LocalizationReActAgent(ReActAgent):
    def __init__(
            self,
            *,
            llm: LLMClient,
            tools: List[BaseTool],
            max_iters: int = 8,
    ):
        system_message = self._build_system_message()
        super().__init__(llm=llm, tools=tools, system_message=system_message, max_iters=max_iters)

    def _build_system_message(self) -> str:
        return LoadPrompt.load_prompt("localization_agent.jinja2", PromptFormat.JINJA2, prompt_type="system").format()

    def _prepare_tool_args(self, tool_name: str, raw_args: Dict, state: AgentState) -> Tuple[str, Dict]:
        if tool_name == "modify_atomic_blocks":
            raw_args = dict(raw_args)
            raw_args.setdefault("current_blocks", getattr(state, "atomic_blocks", []))
        elif tool_name == "finalize_atomic_blocks":
            raw_args = dict(raw_args)
            raw_args.setdefault("current_blocks", getattr(state, "atomic_blocks", []))
        return tool_name, raw_args

    def _process_tool_output(self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List) -> None:
        if tool_call["name"] == "modify_atomic_blocks" and isinstance(result, list) and all(
                isinstance(r, AtomicBlock) for r in result
        ):
            state.atomic_blocks = result
            outputs.append(ToolMessage(content="AtomicBlocks successfully updated.", tool_call_id=tool_call["id"]))
        elif tool_call["name"] == "finalize_atomic_blocks":
            
            try:
                blocks, comments = result
                if isinstance(blocks, list) and all(isinstance(r, AtomicBlock) for r in blocks):
                    state.atomic_blocks = blocks
                    state.final_comments = str(comments)

                    outputs.append(ToolMessage(content=str(comments), tool_call_id=tool_call["id"]))

                    # End the agent
                    setattr(self, "_end_now", True)

                    return
            except Exception:
                pass
            
            outputs.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        else:
            outputs.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
