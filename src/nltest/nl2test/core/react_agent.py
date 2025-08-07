import json
import re
from typing import List, Dict, Any, Union, Optional, Tuple
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, ToolCall
from nltest.nl2test.model.models import AgentState, AtomicBlock
from nltest.utils.llm.format_validator import FormatValidator
from nltest.utils.llm.llm_client import LLMClient


class ReActAgent:
    def __init__(
            self,
            *,
            llm: LLMClient,
            tools: List[BaseTool],
            prompt_template: PromptTemplate,
            nl_description: str,
            **kwargs
    ):
        self.llm = llm
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        self.prompt_template = prompt_template
        self.nl_description = nl_description
        self.checkpointer = kwargs.get("checkpointer") or MemorySaver()
        self.max_iterations = kwargs.get("max_iterations") or 10

        tool_descriptions = [f"**Tool Name**: {tool.name}\n**Tool Description**:\n{tool.description}" for tool in self.tools]
        self.tool_descriptions = "\n\n".join(tool_descriptions)

        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(state_schema=AgentState)
        graph.add_node(node="agent", action=self._agent_node)
        graph.add_node(node="tools", action=self._tools_node)
        graph.add_conditional_edges(source="agent", path=self._route, path_map={"tools": "tools", "end": END})
        graph.add_edge(start_key="tools", end_key="agent")
        graph.set_entry_point(key="agent")
        return graph.compile(checkpointer=self.checkpointer)

    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        if state.iterations >= self.max_iterations:
            # TODO: Maybe make the LLM force-complete
            final_message = AIMessage(content="Max iterations reached. Ending process.")
            return {"messages": state.messages + [final_message], "iterations": state.iterations}

        new_iterations = state.iterations + 1

        instructions = state.messages[0].content if state.messages else ""

        # Compose history
        scratchpad_lines = []
        for message in state.messages[1:]:
            if isinstance(message, AIMessage):
                if message.tool_calls:
                    scratchpad_lines.append(f"Thought: {message.content}")
                    for tc in message.tool_calls:
                        scratchpad_lines.append(f"Tool Name: {tc['name']}")
                        scratchpad_lines.append(f"Tool Input: {tc['args']}")
                else:
                    scratchpad_lines.append(f"Final Answer: {message.content}")
            elif isinstance(message, ToolMessage):
                scratchpad_lines.append(f"Tool Output: {message.content}\n")
        history = "\n".join(scratchpad_lines).strip()

        prompt = self._format_prompt(state, instructions, history)

        response = self.llm.generate(prompt, sanitize=True)

        dict_response = FormatValidator.validate(response, dict)

        parsed_messages = []
        thought = dict_response.get("thought", "")
        final = dict_response.get("final_answer")

        if dict_response.get("tool") and dict_response.get("tool_input"):
            base_len = len(state.messages) + len(parsed_messages)

            # Get stripped tool name
            tool_name = dict_response["tool"].strip()

            # Clean inputs
            tool_input = dict_response["tool_input"]
            tool_input = {
                k: (v.strip() if isinstance(v, str) else v)
                for k, v in tool_input.items()
            }

            tool_call = {
                "id": f"call_{base_len}",
                "name": tool_name,
                "args": tool_input,
                "type": "tool_call"
            }
            parsed_messages.append(AIMessage(content=thought, tool_calls=[tool_call]))
        elif final:
            parsed_messages.append(AIMessage(content=final))
        else:
            parsed_messages.append(AIMessage(content=response))

        new_messages = state.messages + parsed_messages
        return {"messages": new_messages, "iterations": new_iterations}

    def _tools_node(self, state: AgentState) -> Dict[str, Any]:
        last_msg = state.messages[-1]
        # If last message is not a tool call, then we do not process this tool node
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return {"messages": state.messages}

        outputs = []
        for tool_call in last_msg.tool_calls:
            tool_name = tool_call["name"]
            try:
                tool_name, args = self._prepare_tool_args(tool_name, tool_call["args"], state)
            except KeyError:
                outputs.append(ToolMessage(
                    content="Invalid tool arguments format",
                    tool_call_id=tool_call["id"]
                ))
                continue

            # Check if tool exists in map
            if tool_name in self.tool_map:
                try:
                    result = self.tool_map[tool_name].invoke(args)
                    self._process_tool_output(tool_call, result, state, outputs)

                except Exception as e:
                    outputs.append(ToolMessage(
                        content=f"Error executing tool: {str(e)}",
                        tool_call_id=tool_call["id"]
                    ))
            else:
                outputs.append(ToolMessage(
                    content=f"Tool {tool_name} not found",
                    tool_call_id=tool_call["id"]
                ))

        new_messages = state.messages + outputs
        return {"messages": new_messages}

    def _route(self, state: AgentState) -> str:
        last_msg = state.messages[-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "tools"
        return "end"

    def _format_prompt(self, state: AgentState, instructions: str, history: str) -> str:
        raise NotImplementedError()

    def _prepare_tool_args(self, tool_name: str, raw_args: Dict, state: AgentState) -> Tuple[str, Dict]:
        """Implementation for preparing tool call arguments."""
        return tool_name, raw_args

    def _process_tool_output(self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List) -> None:
        """Implementation for processing a tool output. Base implementation is provided."""
        outputs.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))

    def invoke(self, input_msg: str, state: Optional[AgentState] = None, config: Optional[Dict[str, Any]] = None) -> AgentState:
        if state is None:
            effective_state = AgentState(messages=[HumanMessage(content=input_msg)], iterations=0)
        else:
            effective_state = state.model_copy(deep=True)
            effective_state.messages.append(HumanMessage(content=input_msg))
        return self.graph.invoke(effective_state, config=config or {"configurable": {"thread_id": "default"}})
