from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
    ToolCall,
)
from langchain_core.tools import BaseTool
from nltest.nl2test.models import AgentState
from nltest.utils.llm import LLMClient


class ReActAgent:
    def __init__(
            self,
            *,
            llm: LLMClient,
            tools: List[BaseTool],
            allow_duplicate_tools: Optional[List[BaseTool]] = None,
            system_message: Optional[str] = None,
            allow_parallelize: bool = False,
            max_iters: int = 20,
            strict_finalize: bool = True,
            use_checkpointer: bool = True,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.allow_duplicate_tools = allow_duplicate_tools or []
        self.max_iters = max_iters
        self.system_message = system_message or "You are a helpful AI assistant."
        self.allow_parallelize = allow_parallelize
        self.strict_finalize = strict_finalize
        self.use_checkpointer = use_checkpointer

        self._allow_duplicate_tool_names = {t.name for t in self.allow_duplicate_tools}
        self.tool_map: Dict[str, BaseTool] = {t.name: t for t in tools}

        # Internal flag for tool-driven termination
        self._end_now: bool = False

        # Validate finalize tool presence when strict finalize is enabled
        if self.strict_finalize:
            has_finalize = any(getattr(t, "name", None) == "finalize" for t in self.tools)
            if not has_finalize:
                # Defer import to avoid cycles
                from nltest.utils.exceptions import ConfigurationException

                raise ConfigurationException(
                    "finalize_tool_missing",
                    message=(
                        "ReActAgent requires a 'finalize' tool when strict_finalize is enabled."
                    ),
                )

        self.graph = self._build_graph()

    def reset_agent(self) -> None:
        """Reset internal termination flag so the agent can continue running."""
        self._end_now = False

    # Subclass hooks
    def _prepare_tool_args(
            self, tool_name: str, raw_args: Dict[str, Any], state: AgentState
    ) -> Tuple[str, Dict[str, Any]]:
        """Allow subclasses to inject arguments before tool call."""
        return tool_name, raw_args

    def _process_tool_output(
            self,
            tool_call: ToolCall,
            result: Any,
            state: AgentState,
            outputs: List[ToolMessage],
    ) -> None:
        """Allow subclasses to interpret tool results and update state."""
        outputs.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

    def _should_end_after_tools(self, state: AgentState) -> bool:
        """Hook for subclasses to request ending immediately after tools."""
        return getattr(self, "_end_now", False)

    # Graph construction
    def _encode_tool_call(self, tool_name: str, args: Dict[str, Any]) -> str:
        try:
            args_json = json.dumps(
                args, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        except Exception:
            args_json = str(args)
        return f"{tool_name}|{args_json}"

    def _build_graph(self):
        """Organize the graph into nodes and edges. Typical react workflow with model call, tool call, and ending."""
        workflow = StateGraph(AgentState)

        # Call model
        def call_model(state: AgentState) -> AgentState:
            # Ensure a system message starts the conversation; rely on invoke() to set it correctly
            assert state.messages and isinstance(
                state.messages[0], SystemMessage
            ), "First message must be a SystemMessage"

            # Check if we're approaching the iteration limit and add a warning
            remaining_iterations = self.max_iters - state.iterations
            if remaining_iterations == 1:
                warning_message = (
                    "WARNING: This is your last allowed iteration in this sequence. "
                    "You MUST execute the `finalize` tool call now to produce your final output based on all gathered information."
                )
                state.messages.append(HumanMessage(content=warning_message))
            elif remaining_iterations == 2:
                warning_message = (
                    "WARNING: This is your second last allowed iteration in this sequence. "
                    "You MUST execute any last tools to prepare your output. You MUST execute the `finalize` tool after this turn, so perform any last actions before completing as needed."
                )
                state.messages.append(HumanMessage(content=warning_message))

            out: AIMessage = self.llm.invoke_messages(
                state.messages,
                tools=self.tools,
                tool_choice="any",
                extra_model_kwargs={"parallel_tool_calls": self.allow_parallelize},
            )
            state.messages.append(out)
            state.iterations += 1
            return state

        # Conditional edge to determine if the agent should continue
        def should_continue(state: AgentState) -> str:
            # If the last AI message has tool calls, always resolve them first
            last_ai: Optional[AIMessage] = None
            for msg in reversed(state.messages):
                if isinstance(msg, AIMessage):
                    last_ai = msg
                    break

            # If the last message is an AI message and it has tool calls, continue
            if last_ai and getattr(last_ai, "tool_calls", None):
                return "use_tools"

            # No tool calls: if we've reached the limit, transition to force_end
            if state.iterations >= self.max_iters:
                return "force_end"

            return "end"

        # Execute tools if any
        def call_tools(state: AgentState) -> AgentState:
            last_ai: AIMessage = next(
                m for m in reversed(state.messages) if isinstance(m, AIMessage)
            )
            tool_msgs: List[ToolMessage] = []

            for tc in last_ai.tool_calls:
                name: str = tc["name"]
                args: Dict[str, Any] = self.llm.parse_tool_args(tc.get("args"))

                name, args = self._prepare_tool_args(name, args, state)

                tool: BaseTool | None = self.tool_map.get(name)
                if tool is None:
                    tool_msgs.append(
                        ToolMessage(
                            content=f"[Tool '{name}' not found]", tool_call_id=tc["id"]
                        )
                    )
                    continue

                # Duplicate detection and counting per tool/args
                encoding = self._encode_tool_call(name, args)
                tool_history = state.tool_calls.setdefault(name, {})
                prev_count = tool_history.get(encoding, 0)
                tool_history[encoding] = prev_count + 1
                state.curr_tool_trajectory.append(name)

                if prev_count > 0 and name not in self._allow_duplicate_tool_names:
                    # Already executed with identical args; skip and inform the model
                    try:
                        content = json.dumps(
                            {
                                "status": "skipped",
                                "reason": "duplicate_tool_call",
                                "tool": name,
                                "args": args,
                            },
                            ensure_ascii=False,
                        )
                    except Exception:
                        content = (
                            f"skipped: duplicate_tool_call for tool={name} args={args}"
                        )
                    tool_msgs.append(
                        ToolMessage(content=content, tool_call_id=tc["id"])
                    )
                    continue

                try:
                    result = tool.invoke(args)
                except Exception as e:
                    result = f"[{name} raised: {e}]"

                # Let subclasses interpret results + possibly update state
                self._process_tool_output(tc, result, state, tool_msgs)

            state.messages.extend(tool_msgs)
            return state

        # Force the model to end if remaining iterations is 0
        def force_end(state: AgentState) -> AgentState:
            # Ensure a system message starts the conversation
            assert state.messages and isinstance(
                state.messages[0], SystemMessage
            ), "First message must be a SystemMessage"

            # Nudge the model explicitly to call finalize with the available context
            prompt = (
                "You have reached the iteration limit. Use the 'finalize' tool now to produce the final answer "
                "based on all prior tool results and messages. Do not call any other tools or perform any other behavior but finalizing the answer through the tool call."
            )
            human_prompt = HumanMessage(content=prompt)
            # Include the instruction in the state history
            state.messages.append(human_prompt)

            finalize_tools = [t for t in self.tools if t.name == "finalize"]
            tools_to_bind = finalize_tools if finalize_tools else self.tools

            # Route explicitly to the finalize tool when available
            tool_choice = (
                {"type": "tool", "name": "finalize"}
                if finalize_tools
                else "any"
            )

            out: AIMessage = self.llm.invoke_messages(
                state.messages,
                tools=tools_to_bind,
                tool_choice=tool_choice,
                extra_model_kwargs={"parallel_tool_calls": self.allow_parallelize},
            )
            state.messages.append(out)

            # Enter end state so we aren't in infinite force_end loop
            self._end_now = True

            # If strict, ensure finalize was actually called; otherwise raise
            if self.strict_finalize:
                last_ai: Optional[AIMessage] = None
                for msg in reversed(state.messages):
                    if isinstance(msg, AIMessage):
                        last_ai = msg
                        break
                # No tool calls or wrong tool
                if not last_ai or not getattr(last_ai, "tool_calls", None):
                    from nltest.utils.exceptions import ConfigurationException

                    raise ConfigurationException(
                        "finalize_not_called",
                        message=(
                            "Model did not call 'finalize' in force_end despite strict enforcement."
                        ),
                    )
                else:
                    # Validate the tool called is finalize
                    names = [tc.get("name") for tc in last_ai.tool_calls]
                    if not any(n == "finalize" for n in names):
                        from nltest.utils.exceptions import ConfigurationException

                        raise ConfigurationException(
                            "finalize_not_called",
                            message=(
                                "Model called a non-finalize tool during force_end in strict mode."
                            ),
                        )

            return state

        # Decide what to do after force_end model call
        def should_continue_force_end(state: AgentState) -> str:
            last_ai: Optional[AIMessage] = None
            for msg in reversed(state.messages):
                if isinstance(msg, AIMessage):
                    last_ai = msg
                    break
            if last_ai and getattr(last_ai, "tool_calls", None):
                return "use_tools"
            return "end"

        # Decide whether to end after tools or continue/force end
        def should_continue_after_tools(state: AgentState) -> str:
            if self._should_end_after_tools(state):
                return "end"
            # If we've consumed the final allowed model step already, switch to force_end
            if state.iterations >= self.max_iters:
                return "force_end"
            return "continue"

        # Assemble graph: define all nodes and edges together for readability
        workflow.add_node("call_model", call_model)
        workflow.add_node("call_tools", call_tools)
        workflow.add_node("force_end", force_end)

        workflow.set_entry_point("call_model")
        workflow.add_conditional_edges(
            "call_model",
            should_continue,
            {
                "use_tools": "call_tools",
                "force_end": "force_end",
                "end": END,
            },
        )
        workflow.add_conditional_edges(
            "force_end",
            should_continue_force_end,
            {
                "use_tools": "call_tools",
                "end": END,
            },
        )
        workflow.add_conditional_edges(
            "call_tools",
            should_continue_after_tools,
            {
                "continue": "call_model",
                "force_end": "force_end",
                "end": END,
            },
        )

        # Standard in-memory checkpointer (configurable)
        checkpointer = MemorySaver() if getattr(self, "use_checkpointer", True) else None

        return workflow.compile(checkpointer=checkpointer)

    def invoke(
            self,
            input_msg: str,
            state: Optional[AgentState] = None,
            config: Optional[Dict[str, Any]] = None,
    ) -> AgentState:
        self._end_now = False

        if state is None:
            # Fresh state: ensure SystemMessage precedes the HumanMessage
            effective = AgentState(
                messages=[
                    SystemMessage(content=self.system_message),
                    HumanMessage(content=input_msg),
                ],
                iterations=0,
            )
        else:
            effective = (
                state.model_copy(deep=True) if hasattr(state, "model_copy") else state
            )
            # Ensure the first message is a SystemMessage; inject if missing
            if not effective.messages or not isinstance(
                    effective.messages[0], SystemMessage
            ):
                effective.messages.insert(0, SystemMessage(content=self.system_message))

            effective.messages.append(HumanMessage(content=input_msg))

        result = self.graph.invoke(
            effective,
            config=config
                   or {
                       "configurable": {"thread_id": "default"},
                       "recursion_limit": 3 * self.max_iters,
                   },
        )

        # Convert dictionary result back to AgentState if needed
        if isinstance(result, dict):
            return AgentState(**result)
        return result
