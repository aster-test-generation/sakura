from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Type

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from sakura.nl2test.models import AgentState
from sakura.utils.exceptions import ConfigurationException
from sakura.utils.llm import LLMClient
from sakura.utils.pretty.color_logger import RichLog
from sakura.utils.tool_messages import (
    format_tool_error,
    format_tool_ok,
)


class ReActAgent:
    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: List[BaseTool],
        allow_duplicate_tools: Optional[List[BaseTool]] = None,
        system_message: Optional[str] = None,
        allow_parallelize: bool = True,
        max_iters: int = 20,
        strict_finalize: bool = True,
        use_checkpointer: bool = True,
        max_force_end_attempts: int = 3,
        max_no_tool_retries: int = 2,
    ) -> None:
        self.llm: LLMClient = llm
        self.tools = tools
        self.allow_duplicate_tools = allow_duplicate_tools or []
        self.max_iters = max_iters
        self.system_message = system_message or "You are a helpful AI assistant."
        self.allow_parallelize = allow_parallelize
        self.strict_finalize = strict_finalize
        self.use_checkpointer = use_checkpointer
        self.max_force_end_attempts = max_force_end_attempts
        self.max_no_tool_retries = max_no_tool_retries

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

    def _get_finalize_schema(self) -> Type[BaseModel]:
        """Return the Pydantic schema for finalize args. Must be implemented by subclasses."""
        raise NotImplementedError(
            "Subclasses must implement _get_finalize_schema() to return a Pydantic schema."
        )

    def _get_force_finalize_system_prompt(self) -> str:
        """Return system prompt for force_finalize structured output. Must be implemented by subclasses."""
        raise NotImplementedError(
            "Subclasses must implement _get_force_finalize_system_prompt() to return a system prompt."
        )

    def _get_force_finalize_chat_prompt(self) -> str:
        """Return chat prompt for force_finalize structured output. Must be implemented by subclasses."""
        raise NotImplementedError(
            "Subclasses must implement _get_force_finalize_chat_prompt() to return a chat prompt."
        )

    def _process_force_finalize_result(
        self, result: BaseModel, state: AgentState
    ) -> None:
        """Process the structured finalize result. Must be implemented by subclasses."""
        raise NotImplementedError(
            "Subclasses must implement _process_force_finalize_result() to process the result."
        )

    def _execute_force_end(self, state: AgentState) -> AgentState:
        """Execute force finalize logic. Override in subclasses to customize behavior."""
        state.force_end_attempts += 1

        finalize_schema = self._get_finalize_schema()
        force_system = self._get_force_finalize_system_prompt()
        force_chat = self._get_force_finalize_chat_prompt()
        messages_for_structured: List[BaseMessage] = [
            SystemMessage(content=force_system),
        ]
        for m in state.messages:
            if isinstance(m, BaseMessage) and not isinstance(m, SystemMessage):
                messages_for_structured.append(m)

        # Insert AIMessage before HumanMessage if last message is ToolMessage
        if messages_for_structured and isinstance(
            messages_for_structured[-1], ToolMessage
        ):
            messages_for_structured.append(AIMessage(content="Acknowledged."))

        messages_for_structured.append(HumanMessage(content=force_chat))

        try:
            result = self.llm.invoke_structured_with_retries(
                messages=messages_for_structured,
                schema=finalize_schema,
                strict=True,
                max_attempts=self.max_force_end_attempts,
                on_failure="return_none",
            )
        except Exception as exc:
            RichLog.warn(f"[ReActAgent] force_end structured output failed: {exc}")
            result = None

        if result is not None:
            self._process_force_finalize_result(result, state)
            self._log_force_finalize(state)
            return state

        if self.strict_finalize:
            raise ConfigurationException(
                "finalize_not_called",
                message="Failed to produce finalize output after max attempts.",
                details={"attempts": self.max_force_end_attempts},
            )
        state.finalize_called = True
        state.final_comments = "Auto-finalized: structured output failed"
        self._log_force_finalize(state)
        return state

    def _log_force_finalize(self, state: AgentState) -> None:
        """Log finalize to tool trajectory and counts for force_end consistency."""
        state.curr_tool_trajectory.append("finalize")
        total_history = state.total_tool_calls.setdefault("finalize", {})
        total_history["<force_finalize>"] = total_history.get("<force_finalize>", 0) + 1

    # Subclass hook
    def prepare_tool_args(
        self, tool_name: str, raw_args: Dict[str, Any], state: AgentState
    ) -> Tuple[str, Dict[str, Any]]:
        """Hook for subclasses to inject or transform tool arguments before execution."""
        return tool_name, raw_args

    def _append_tool_error_if_needed(
        self,
        tool_call: ToolCall,
        result: Any,
        outputs: List[ToolMessage],
    ) -> bool:
        if isinstance(result, dict) and result.get("status") == "error":
            error_info = result.get("error") or {}
            details = dict(error_info.get("details") or {})
            details.setdefault("tool", tool_call.get("name"))
            details.setdefault("tool_call_id", tool_call.get("id"))
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code=error_info.get("code") or "tool_error",
                        message=error_info.get("message") or "Tool error",
                        details=details,
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
            return True
        return False

    def _append_tool_output(
        self,
        tool_call: ToolCall,
        result: Any,
        outputs: List[ToolMessage],
    ) -> None:
        if self._append_tool_error_if_needed(tool_call, result, outputs):
            return
        outputs.append(
            ToolMessage(
                content=format_tool_ok(result),
                tool_call_id=tool_call["id"],
            )
        )

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
            self._append_tool_output(tool_call, result, outputs)
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
        tool_call_id = details.get("tool_call_id") or getattr(
            tool_msg, "tool_call_id", "unknown_call"
        )
        error_code = error_info.get("code") or "unknown_error"
        error_msg = error_info.get("message") or ""

        matching_call: ToolCall | None = None
        tool_calls = llm_message.tool_calls or []
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
                        segments.append(
                            str(chunk.get("text") or chunk.get("content") or chunk)
                        )
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
            assert state.messages and isinstance(state.messages[0], SystemMessage), (
                "First message must be a SystemMessage"
            )

            # Check if we're approaching the iteration limit and prepare a warning
            remaining = self.max_iters - state.iterations
            notice = None
            if remaining == 1:
                notice = (
                    "SYSTEM NOTICE: Final iteration. "
                    "Call the finalize tool now using the best available context."
                )
            elif remaining == 2:
                notice = (
                    "SYSTEM NOTICE: Second-to-last iteration. "
                    "Finish any remaining tool work now; plan to call finalize next turn."
                )

            messages_for_llm = list(state.messages)
            if notice:
                last_msg = messages_for_llm[-1]
                if isinstance(last_msg, ToolMessage):
                    messages_for_llm.append(AIMessage(content="Acknowledged."))
                    messages_for_llm.append(HumanMessage(content=notice))
                elif isinstance(last_msg, HumanMessage):
                    # Consolidate with existing HumanMessage to avoid consecutive HumanMessages
                    existing_content = (
                        last_msg.content
                        if isinstance(last_msg.content, str)
                        else str(last_msg.content)
                    )
                    messages_for_llm[-1] = HumanMessage(
                        content=f"{existing_content}\n\n{notice}"
                    )
                else:
                    # Last message is AIMessage - just append HumanMessage
                    messages_for_llm.append(HumanMessage(content=notice))

            out: AIMessage = self.llm.invoke_messages(
                messages_for_llm,
                tools=self.tools,
                tool_choice="auto",
                extra_model_kwargs={"parallel_tool_calls": self.allow_parallelize},
                context={
                    "agent": type(self).__name__,
                    "iteration": state.iterations,
                    "max_iters": self.max_iters,
                    "no_tool_retries": state.no_tool_retries,
                    "finalize_called": state.finalize_called,
                },
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
            tool_calls = last_ai.tool_calls or []
            tool_names = [tc.get("name") for tc in tool_calls]
            RichLog.debug(
                f"[ReActAgent] call_tools: iteration={state.iterations}, "
                f"tools={tool_names}"
            )
            # Reset no-tool retry counter since we have tool calls
            state.no_tool_retries = 0
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
                curr_history = state.curr_tool_calls.setdefault(name, {})
                prev_count = curr_history.get(encoding, 0)
                curr_history[encoding] = prev_count + 1

                total_history = state.total_tool_calls.setdefault(name, {})
                total_prev_count = total_history.get(encoding, 0)
                total_history[encoding] = total_prev_count + 1
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

        # Nudge the model to call a tool when it returns empty tool calls
        def nudge_model(state: AgentState) -> AgentState:
            nudge_message = HumanMessage(
                content=(
                    "You must call a tool to proceed. If you have completed your task, "
                    "call the finalize tool. Otherwise, call an appropriate tool to continue."
                )
            )
            state.messages.append(nudge_message)
            return state

        # Force the model to end if remaining iterations is 0
        def force_end(state: AgentState) -> AgentState:
            RichLog.debug(
                f"[ReActAgent] force_end: triggering force finalize "
                f"(iteration={state.iterations}, attempts={state.force_end_attempts})"
            )
            return self._execute_force_end(state)

        # Decide whether to end after tools or continue/force end
        def should_continue_after_tools(state: AgentState) -> str:
            end_now = self._should_end_after_tools(state)
            if end_now:
                RichLog.debug(
                    f"[ReActAgent] should_continue_after_tools: 'end' "
                    f"(_end_now=True, iteration={state.iterations})"
                )
                return "end"
            if state.iterations >= self.max_iters:
                RichLog.debug(
                    f"[ReActAgent] should_continue_after_tools: 'force_end' "
                    f"(iteration={state.iterations} >= max_iters={self.max_iters})"
                )
                return "force_end"
            RichLog.debug(
                f"[ReActAgent] should_continue_after_tools: 'continue' "
                f"(iteration={state.iterations})"
            )
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
                tool_names = [tc.get("name") for tc in last_ai.tool_calls]
                RichLog.debug(
                    f"[ReActAgent] should_continue_after_llm: 'use_tools' "
                    f"(tools={tool_names}, iteration={state.iterations})"
                )
                return "use_tools"

            # No tool calls: check if we should retry before forcing end
            if self.strict_finalize and not state.finalize_called:
                # Give the model another chance if we haven't exceeded retry limit
                if state.no_tool_retries < self.max_no_tool_retries:
                    state.no_tool_retries += 1
                    RichLog.debug(
                        f"[ReActAgent] should_continue_after_llm: 'nudge' "
                        f"(no tool calls, retry {state.no_tool_retries}/"
                        f"{self.max_no_tool_retries}, iteration={state.iterations})"
                    )
                    return "nudge"
                RichLog.debug(
                    f"[ReActAgent] should_continue_after_llm: 'force_end' "
                    f"(strict_finalize=True, finalize_called=False, "
                    f"retries exhausted={state.no_tool_retries}, "
                    f"iteration={state.iterations})"
                )
                return "force_end"

            if state.iterations >= self.max_iters:
                RichLog.debug(
                    f"[ReActAgent] should_continue_after_llm: 'force_end' "
                    f"(iteration={state.iterations} >= max_iters={self.max_iters})"
                )
                return "force_end"

            RichLog.debug(
                f"[ReActAgent] should_continue_after_llm: 'end' "
                f"(finalize_called={state.finalize_called}, iteration={state.iterations})"
            )
            return "end"

        # Assemble graph: define all nodes and edges together for readability
        workflow.add_node("call_model", call_model)
        workflow.add_node("call_tools", call_tools)
        workflow.add_node("nudge_model", nudge_model)
        workflow.add_node("force_end", force_end)

        workflow.set_entry_point("call_model")
        workflow.add_conditional_edges(
            "call_model",
            should_continue_after_llm,
            {
                "use_tools": "call_tools",
                "nudge": "nudge_model",
                "force_end": "force_end",
                "end": END,
            },
        )
        workflow.add_edge("nudge_model", "call_model")
        workflow.add_edge("force_end", END)
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

        # Multiplier accounts for potential nudge cycles (call_model -> nudge -> call_model -> tools)
        min_recursion_limit = 5 * self.max_iters
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
