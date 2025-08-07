from typing import List, Any, Dict, Tuple

from langchain_core.messages import ToolMessage, ToolCall
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from nltest.nl2test.core.react_agent import ReActAgent
from nltest.nl2test.model.models import AgentState
from nltest.utils.llm.llm_client import LLMClient


class CompositionReActAgent(ReActAgent):
    def __init__(
            self,
            *,
            llm: LLMClient,
            tools: List[BaseTool],
            prompt_template: PromptTemplate,
            nl_description: str,
            **kwargs,
    ):
        super().__init__(
            llm=llm,
            tools=tools,
            prompt_template=prompt_template,
            nl_description=nl_description,
            kwargs=kwargs
        )

    def _format_prompt(self, state: AgentState, instructions: str, history: str) -> str:
        prompt = self.prompt_template.format(
            instructions=instructions,
            nl_description=self.nl_description,
            atomic_blocks=state.atomic_blocks,
            tool_descriptions=self.tool_descriptions,
            history=history,
        )
        return prompt

    def _prepare_tool_args(self, tool_name: str, raw_args: Dict, state: AgentState) -> Tuple[str, Dict]:
        """Implementation for preparing tool call arguments."""
        return tool_name, raw_args

    def _process_tool_output(self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List) -> None:
        """Implementation for processing a tool output."""
        outputs.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))
