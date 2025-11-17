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
from nltest.utils.tool_messages import (
    format_tool_error,
    format_tool_ok,
)
from nltest.utils.pretty.color_logger import RichLog
from nltest.utils.exceptions import ConfigurationException


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
            has_finalize = any(tool.name == "finalize" for tool in self.tools)
            if not has_finalize:
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

    # Subclass hook
    def prepare_tool_args(
            self, tool_name: str, raw_args: Dict[str, Any], state: AgentState
    ) -> Tuple[str, Dict[str, Any]]:
        """Hook for subclasses to inject or transform tool arguments before execution."""
        return tool_name, raw_args

    # Subclass hook
    def process_tool_output(
            self,
            tool_call: ToolCall,
            result: Any,
            state: AgentState,
            outputs: List[ToolMessage],
    ) -> None:
        """Allow subclasses to interpret tool results and update state."""
        try:
            outputs.append(
                ToolMessage(
                    content=format_tool_ok(result),
                    tool_call_id=tool_call["id"],
                )
            )
        except Exception as exc:
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code=type(exc).__name__,
                        message=str(exc),
                        details={
                            "tool": tool_call.get("name"),
                            "tool_call_id": tool_call["id"],
                        },
                    ),
                    tool_call_id=tool_call["id"],
                )
            )

    # Subclass hook
    def process_llm_output(self, tool_call: ToolCall, state: AgentState) -> None:
        pass

    def _should_end_after_tools(self, state: AgentState) -> bool:
        """Hook for subclasses to request ending immediately after tools."""
        return self._end_now

    # Graph construction
    def _encode_tool_call(self, tool_name: str, args: Dict[str, Any]) -> str:
        try:
            args_json = json.dumps(
                args, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        except Exception:
            args_json = str(args)
        return f"{tool_name}|{args_json}"

    # For debugging
    def _log_tool_error(self, tool_msg: ToolMessage, llm_message: AIMessage) -> None:
        """Emit a RichLog debug entry containing the tool error and triggering LLM turn."""

        content = tool_msg.content
        if isinstance(content, str):
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                return
        elif isinstance(content, dict):
            payload = content
        else:
            return

        if payload.get("status") != "error":
            return

        error_info = payload.get("error") or {}
        details = error_info.get("details") or {}
        tool_name = details.get("tool") or "unknown_tool"
        tool_call_id = details.get("tool_call_id") or getattr(tool_msg, "tool_call_id", "unknown_call")
        error_code = error_info.get("code") or "unknown_error"
        error_msg = error_info.get("message") or ""

        matching_call: ToolCall | None = None
        tool_calls = llm_message.tool_calls
        for tc in tool_calls:
            if tc.get("id") == tool_call_id:
                matching_call = tc
                break

        if matching_call:
            llm_info_label = "llm_tool_args"
            tool_name = matching_call.get("name") or tool_name
            raw_args = matching_call.get("args")
            parsed_args = self.llm.parse_tool_args(raw_args)
            try:
                if parsed_args:
                    llm_info_value = json.dumps(parsed_args, ensure_ascii=False)
                elif isinstance(raw_args, str):
                    llm_info_value = raw_args
                else:
                    llm_info_value = str(raw_args)
            except Exception:
                llm_info_value = str(parsed_args or raw_args)
        else:  # Fallback is generic LLM content message (if tool can't be matched)
            llm_info_label = "llm_content"
            llm_content = llm_message.content
            if isinstance(llm_content, str):
                llm_text = llm_content
            elif isinstance(llm_content, list):
                segments = []
                for chunk in llm_content:
                    if isinstance(chunk, dict):
                        segments.append(str(chunk.get("text") or chunk.get("content") or chunk))
                    else:
                        segments.append(str(chunk))
                llm_text = "\n".join(segments)
            else:
                llm_text = str(llm_content)

            try:
                sanitized = self.llm.sanitize(llm_text)
            except Exception:
                sanitized = llm_text

            llm_info_value = sanitized

        llm_preview = (llm_info_value or "")[:500]
        RichLog.debug(
            f"[ReActAgent] tool_error name={tool_name} call_id={tool_call_id} "
            f"code={error_code} message={error_msg} | {llm_info_label}={llm_preview}"
        )

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
                    "SYSTEM NOTICE: Final iteration. Call the finalize tool now using the best available context."
                )
                state.messages.append(HumanMessage(content=warning_message))
            elif remaining_iterations == 2:
                warning_message = (
                    "SYSTEM NOTICE: Second-to-last iteration. Finish any remaining tool work now; plan to call finalize next turn."
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

        # Execute tools if any
        def call_tools(state: AgentState) -> AgentState:
            last_ai: AIMessage = next(
                m for m in reversed(state.messages) if isinstance(m, AIMessage)
            )
            tool_msgs: List[ToolMessage] = []
            tool_calls = last_ai.tool_calls
            skipped_tool_calls: List[ToolCall] = []
            if not self.allow_parallelize and len(tool_calls) > 1:
                skipped_tool_calls = tool_calls[1:]
                tool_calls = tool_calls[:1]

            for tc in tool_calls:
                name: str = tc.get("name")
                args: Dict[str, Any] = self.llm.parse_tool_args(tc.get("args"))

                name, args = self.prepare_tool_args(name, args, state)

                tool: BaseTool | None = self.tool_map.get(name)
                if tool is None:
                    tool_msgs.append(
                        ToolMessage(
                            content=format_tool_error(
                                code="tool_not_found",
                                message=f"Tool '{name}' is not registered.",
                                details={
                                    "tool": name,
                                    "tool_call_id": tc["id"],
                                },
                            ),
                            tool_call_id=tc["id"],
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
                    tool_msgs.append(
                        ToolMessage(
                            content=format_tool_error(
                                code="duplicate_tool_call",
                                message="Skipped duplicate tool call with identical arguments.",
                                details={
                                    "tool": name,
                                    "args": args,
                                    "tool_call_id": tc["id"],
                                },
                            ),
                            tool_call_id=tc["id"],
                        )
                    )
                    continue

                try:
                    result = tool.invoke(args)
                except Exception as exc:
                    tool_msgs.append(
                        ToolMessage(
                            content=format_tool_error(
                                code=type(exc).__name__,
                                message=str(exc),
                                details={
                                    "tool": name,
                                    "tool_call_id": tc["id"],
                                    "args": args,
                                },
                            ),
                            tool_call_id=tc["id"],
                        )
                    )
                    continue

                # Let subclasses interpret results + possibly update state
                self.process_tool_output(tc, result, state, tool_msgs)
                self.process_llm_output(tc, state)

            for skipped_tc in skipped_tool_calls:
                skipped_args = self.llm.parse_tool_args(skipped_tc.get("args"))
                tool_msgs.append(
                    ToolMessage(
                        content=format_tool_error(
                            code="parallel_call_disallowed",
                            message=(
                                "Skipped additional tool call because only one tool is allowed per iteration. "
                                "Resend the tool call in a new turn."
                            ),
                            details={
                                "tool": skipped_tc.get("name"),
                                "tool_call_id": skipped_tc["id"],
                                "args": skipped_args,
                            },
                        ),
                        tool_call_id=skipped_tc["id"],
                    )
                )

            for msg in tool_msgs:
                self._log_tool_error(msg, last_ai)

            state.messages.extend(tool_msgs)
            return state

        # Force the model to end if remaining iterations is 0
        def force_end(state: AgentState) -> AgentState:
            state.force_end_attempts += 1
            attempt = state.force_end_attempts

            # Nudge the model explicitly to call finalize with the available context
            if attempt == 1:
                prompt = (
                    "You have reached the iteration limit. Use the 'finalize' tool now to produce the final answer "
                    "based on all prior tool results and messages. Do not call any other tools or perform any other behavior."
                )
            else:
                prompt = (
                    "FINAL NOTICE: You must call the 'finalize' tool immediately. No other tools are allowed. "
                    "Summarize the best available result, then call finalize now."
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
                extra_model_kwargs={"parallel_tool_calls": False},  # No parallel tool call; force end
            )
            state.messages.append(out)

            last_ai: Optional[AIMessage] = out
            tool_calls: List[ToolCall] = last_ai.tool_calls
            requested_finalize = any(tc.get("name") == "finalize" for tc in tool_calls)

            if self.strict_finalize:
                if not tool_calls or not requested_finalize:
                    raise ConfigurationException(
                        "finalize_not_called",
                        message=(
                            "Model did not call 'finalize' in force_end despite strict enforcement."
                            if not tool_calls
                            else "Model called a non-finalize tool during force_end in strict mode."
                        ),
                        details={"state": state},
                    )
            else:
                if not requested_finalize:
                    # Prevent execution of unrelated tools; retry with stronger reminder.
                    last_ai.tool_calls = []
                    if attempt >= 2:
                        raise ConfigurationException(
                            "non_strict_finalize_not_called",
                            message=(
                                "Model failed to call 'finalize' after explicit reminders; aborting."
                            ),
                            details={"attempts": attempt, "state": state},
                        )

            return state

        # Decide what to do after force_end model call
        def should_continue_force_end(state: AgentState) -> str:
            last_ai: Optional[AIMessage] = None
            for msg in reversed(state.messages):
                if isinstance(msg, AIMessage):
                    last_ai = msg
                    break
            if last_ai and last_ai.tool_calls:
                return "use_tools"
            if (
                    not state.finalize_called
                    and not self.strict_finalize
                    and state.force_end_attempts < 2
            ):
                return "force_end"
            return "end"

        # Decide whether to end after tools or continue/force end
        def should_continue_after_tools(state: AgentState) -> str:
            if self._should_end_after_tools(state):  # If the end_now is called from finalize tool
                return "end"
            # If we've consumed the final allowed model step already, switch to force_end
            if state.iterations >= self.max_iters:
                return "force_end"
            return "continue"

        # Conditional edge to determine if the agent should continue
        def should_continue_after_llm(state: AgentState) -> str:
            # If the last AI message has tool calls, always resolve them first
            last_ai: Optional[AIMessage] = None
            for msg in reversed(state.messages):
                if isinstance(msg, AIMessage):
                    last_ai = msg
                    break

            # If the last message is an AI message and it has tool calls, continue
            if last_ai and last_ai.tool_calls:
                return "use_tools"

            # No tool calls: enforce finalize when required and guard iteration limit
            if self.strict_finalize and not state.finalize_called:
                return "force_end"

            if state.iterations >= self.max_iters:
                return "force_end"

            return "end"

        # Assemble graph: define all nodes and edges together for readability
        workflow.add_node("call_model", call_model)
        workflow.add_node("call_tools", call_tools)
        workflow.add_node("force_end", force_end)

        workflow.set_entry_point("call_model")
        workflow.add_conditional_edges(
            "call_model",
            should_continue_after_llm,
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
        checkpointer = MemorySaver() if self.use_checkpointer else None

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

        min_recursion_limit = 3 * self.max_iters
        if config is None:
            final_config: Dict[str, Any] = {
                "configurable": {"thread_id": "default"},
                "recursion_limit": min_recursion_limit,
            }
        else:
            final_config = dict(config)
            current_limit = final_config.get("recursion_limit")
            if current_limit is None or current_limit < min_recursion_limit:
                final_config["recursion_limit"] = min_recursion_limit

        result = self.graph.invoke(
            effective,
            config=final_config,
        )

        # Convert dictionary result back to AgentState if needed
        if isinstance(result, dict):
            return AgentState(**result)
        return result
