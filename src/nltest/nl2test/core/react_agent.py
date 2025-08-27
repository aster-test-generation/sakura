from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage, ToolMessage, ToolCall
from langchain_core.tools import BaseTool

# These imports exist in your repo; we keep them to avoid breaking callers.
from nltest.nl2test.model.models import AgentState
from nltest.utils.llm import LLMClient


class ReActAgent:
    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: List[BaseTool],
        system_message: Optional[str] = None,
        max_iters: int = 20,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_iters = max_iters
        self.system_message = system_message or "You are a helpful AI assistant."

        self.tool_map: Dict[str, BaseTool] = {t.name: t for t in tools}
        
        # Internal flag for tool-driven termination
        self._end_now: bool = False

        self.graph = self._build_graph()

    # Subclass hooks
    def _prepare_tool_args(self, tool_name: str, raw_args: Dict[str, Any], state: AgentState) -> Tuple[str, Dict[str, Any]]:
        """Allow subclasses to inject arguments before tool call."""
        return tool_name, raw_args

    def _process_tool_output(self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List[ToolMessage]) -> None:
        """Allow subclasses to interpret tool results and update state."""
        outputs.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

    def _should_end_after_tools(self, state: AgentState) -> bool:
        """Hook for subclasses to request ending immediately after tools."""
        return getattr(self, "_end_now", False)

    # Graph construction
    def _build_graph(self):
        """Organize the graph into nodes and edges. Typical react workflow with model call, tool call, and ending."""
        workflow = StateGraph(AgentState)

        # Call model
        def call_model(state: AgentState) -> AgentState:
            messages: List[BaseMessage] = list(state.messages)

            # Catch if system message is not present
            if not messages or messages[0].type != "system":
                messages = [SystemMessage(content=self.system_message), *messages]

            # Check if we're approaching the iteration limit and add a warning
            remaining_iterations = self.max_iters - state.iterations
            if remaining_iterations <= 4 and remaining_iterations > 0:
                warning_message = f"WARNING: You have only {remaining_iterations} more iteration(s) allowed in this sequence. Use all the information you have gathered so far to generate and complete your final result immediately. Call the necessary tools to update your state if needed, and then provide your final answer."
                messages.append(SystemMessage(content=warning_message))

            out: AIMessage = self.llm.invoke_messages(
                messages,
                tools=self.tools,
                tool_choice="any",
                # extra_model_kwargs={"parallel_tool_calls": False} # NOTE: Try and enforce only one tool call at a time
            )
            state.messages.append(out)
            state.iterations += 1
            return state

        # Conditional edge to determine if the agent should continue
        def should_continue(state: AgentState) -> str:
            if state.iterations >= self.max_iters:
                return "end"
            
            last_ai: Optional[AIMessage] = None
            for msg in reversed(state.messages):
                if isinstance(msg, AIMessage):
                    last_ai = msg
                    break

            # If the last message is an AI message and it has tool calls, continue
            if last_ai and getattr(last_ai, "tool_calls", None):
                return "use_tools"

            return "end"

        # Execute tools if any
        def call_tools(state: AgentState) -> AgentState:
            last_ai: AIMessage = next(m for m in reversed(state.messages) if isinstance(m, AIMessage))
            tool_msgs: List[ToolMessage] = []

            for tc in last_ai.tool_calls:
                name: str = tc["name"]
                args: Dict[str, Any] = self.llm.parse_tool_args(tc.get("args"))

                # Allow subclasses to tweak the arguments using state -> Mostly for injecting the atomic blocks for modification prompts
                name, args = self._prepare_tool_args(name, args, state)

                tool: BaseTool | None = self.tool_map.get(name)
                if tool is None:
                    tool_msgs.append(ToolMessage(
                        content=f"[Tool '{name}' not found]",
                        tool_call_id=tc["id"]
                    ))
                    continue

                try:
                    result = tool.invoke(args)
                except Exception as e:
                    result = f"[{name} raised: {e}]"

                # Let subclasses interpret results + possibly update state
                self._process_tool_output(tc, result, state, tool_msgs)

            state.messages.extend(tool_msgs)
            return state

        # Assemble graph
        workflow.add_node("call_model", call_model)
        workflow.add_node("call_tools", call_tools)

        workflow.set_entry_point("call_model")
        workflow.add_conditional_edges(
            "call_model",
            should_continue,
            {
                "use_tools": "call_tools",
                "end": END,
            },
        )

        # Decide whether to end after tools or continue
        def should_continue_after_tools(state: AgentState) -> str:
            return "end" if self._should_end_after_tools(state) else "continue"

        workflow.add_conditional_edges(
            "call_tools",
            should_continue_after_tools,
            {
                "continue": "call_model",
                "end": END,
            },
        )

        # Standard in-memory checkpointer
        checkpointer = MemorySaver()

        return workflow.compile(checkpointer=checkpointer)

    def invoke(self, input_msg: str, state: Optional[AgentState] = None, config: Optional[Dict[str, Any]] = None) -> AgentState:
        self._end_now = False
        
        if state is None:
            effective = AgentState(messages=[HumanMessage(content=input_msg)], iterations=0)
        else:
            effective = state.model_copy(deep=True) if hasattr(state, "model_copy") else state
            effective.messages.append(HumanMessage(content=input_msg))

        result = self.graph.invoke(effective, config=config or {"configurable": {"thread_id": "default"}, "recursion_limit": 60})
        
        # Convert dictionary result back to AgentState if needed
        if isinstance(result, dict):
            return AgentState(**result)
        return result
