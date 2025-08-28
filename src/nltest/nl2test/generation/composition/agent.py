from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.messages import ToolMessage, ToolCall
from langchain_core.tools import BaseTool

from nltest.nl2test.core.react_agent import ReActAgent
from nltest.nl2test.models import AgentState
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.utils.llm.llm_client import LLMClient


class CompositionReActAgent(ReActAgent):
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
        return LoadPrompt.load_prompt("composition_agent.jinja2", PromptFormat.JINJA2, prompt_type="system").format()

    def _prepare_tool_args(self, tool_name: str, raw_args: Dict, state: AgentState) -> Tuple[str, Dict]:
        """Implementation for preparing tool call arguments."""
        return tool_name, raw_args

    def _process_tool_output(self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List) -> None:
        """Implementation for processing a tool output."""
        outputs.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))
