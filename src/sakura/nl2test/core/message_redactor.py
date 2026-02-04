from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.messages import AIMessage, ToolMessage

from sakura.nl2test.models import AgentState


class MessageRedactor:
    """
    Shared utility for redacting stale tool outputs and inputs from agent message history.
    Used to reduce token usage by replacing outdated data with placeholders.
    """

    @staticmethod
    def _load_payload(content: Any) -> Tuple[Optional[Dict[str, Any]], str]:
        """Load JSON payload from message content, returning (payload, type_hint)."""
        if isinstance(content, str):
            try:
                return json.loads(content), "str"
            except json.JSONDecodeError:
                return None, "str"
        if isinstance(content, dict):
            return content, "dict"
        return None, "other"

    @staticmethod
    def _store_payload(
        message: ToolMessage, payload: Dict[str, Any], payload_type: str
    ) -> None:
        """Store payload back into message content with appropriate format."""
        if payload_type == "str":
            message.content = json.dumps(
                payload, separators=(",", ":"), ensure_ascii=False
            )
        else:
            message.content = payload

    @staticmethod
    def _build_call_lookup(state: AgentState) -> Dict[str, str]:
        """Build mapping from tool_call_id to tool_name from AIMessages."""
        call_lookup: Dict[str, str] = {}
        for message in state.messages:
            if isinstance(message, AIMessage):
                tool_calls = message.tool_calls or []
                for tc in tool_calls:
                    tc_id = tc.get("id")
                    tc_name = tc.get("name")
                    if tc_id and tc_name:
                        call_lookup[tc_id] = tc_name
        return call_lookup

    @staticmethod
    def redact_tool_outputs(
        state: AgentState,
        tool_names: Set[str],
        keys_to_redact: Dict[str, str],
        status_filter: str = "ok",
    ) -> None:
        """
        Redact specific keys in tool output messages for given tool names.

        Args:
            state: The agent state containing messages to redact
            tool_names: Set of tool names whose outputs should be redacted
            keys_to_redact: Dict mapping data keys to their placeholder strings
            status_filter: Only redact messages with this status (default "ok")
        """
        call_lookup = MessageRedactor._build_call_lookup(state)

        for message in state.messages:
            if not isinstance(message, ToolMessage):
                continue

            tool_call_id = message.tool_call_id
            tool_name = message.name
            if not tool_name and tool_call_id:
                tool_name = call_lookup.get(tool_call_id)

            if tool_name not in tool_names:
                continue

            payload, payload_type = MessageRedactor._load_payload(message.content)
            if not isinstance(payload, dict):
                continue

            if payload.get("status") != status_filter:
                continue

            data = payload.get("data")
            if not isinstance(data, dict):
                continue

            updated = False
            for key, placeholder in keys_to_redact.items():
                if key in data:
                    data[key] = placeholder
                    updated = True

            if updated:
                MessageRedactor._store_payload(message, payload, payload_type)

    @staticmethod
    def redact_tool_outputs_by_tool(
        state: AgentState,
        tool_redactions: Dict[str, Dict[str, str]],
        status_filter: str = "ok",
    ) -> None:
        """
        Redact tool outputs with tool-specific key mappings.

        Args:
            state: The agent state containing messages to redact
            tool_redactions: Dict mapping tool_name -> {key: placeholder}
            status_filter: Only redact messages with this status (default "ok")
        """
        call_lookup = MessageRedactor._build_call_lookup(state)

        for message in state.messages:
            if not isinstance(message, ToolMessage):
                continue

            tool_call_id = message.tool_call_id
            tool_name = message.name
            if not tool_name and tool_call_id:
                tool_name = call_lookup.get(tool_call_id)

            if tool_name not in tool_redactions:
                continue

            payload, payload_type = MessageRedactor._load_payload(message.content)
            if not isinstance(payload, dict):
                continue

            if payload.get("status") != status_filter:
                continue

            data = payload.get("data")
            if not isinstance(data, dict):
                continue

            keys_to_redact = tool_redactions[tool_name]
            updated = False
            for key, placeholder in keys_to_redact.items():
                if key in data:
                    data[key] = placeholder
                    updated = True

            if updated:
                MessageRedactor._store_payload(message, payload, payload_type)

    @staticmethod
    def redact_tool_inputs(
        state: AgentState,
        tool_name: str,
        keys_to_redact: Dict[str, str],
        llm_client: Any,
        exclude_latest: bool = True,
    ) -> None:
        """
        Redact specific keys in tool call arguments for a given tool name.

        Args:
            state: The agent state containing messages to redact
            tool_name: Name of the tool whose inputs should be redacted
            keys_to_redact: Dict mapping argument keys to their placeholder strings
            llm_client: LLM client with parse_tool_args method
            exclude_latest: If True, don't redact the most recent call (default True)
        """
        latest_call_id: Optional[str] = None

        if exclude_latest:
            for message in reversed(state.messages):
                if not isinstance(message, AIMessage):
                    continue
                tool_calls = message.tool_calls or []
                for call in reversed(tool_calls):
                    if call.get("name") == tool_name:
                        latest_call_id = call.get("id")
                        break
                if latest_call_id:
                    break

        for message in state.messages:
            if not isinstance(message, AIMessage):
                continue

            tool_calls = message.tool_calls or []
            for idx, call in enumerate(tool_calls):
                if call.get("name") != tool_name:
                    continue
                if exclude_latest and call.get("id") == latest_call_id:
                    continue

                raw_args = call.get("args")
                parsed_args = llm_client.parse_tool_args(raw_args)

                if isinstance(parsed_args, dict):
                    sanitized_args = dict(parsed_args)
                elif isinstance(raw_args, dict):
                    sanitized_args = dict(raw_args)
                else:
                    sanitized_args = {}

                updated = False
                for key, placeholder in keys_to_redact.items():
                    if key in sanitized_args:
                        sanitized_args[key] = placeholder
                        updated = True

                if not updated:
                    continue

                try:
                    if isinstance(raw_args, str):
                        call["args"] = json.dumps(
                            sanitized_args, ensure_ascii=True, sort_keys=True
                        )
                    else:
                        call["args"] = sanitized_args
                except Exception:
                    call["args"] = sanitized_args

                tool_calls[idx] = call

                additional_kwargs = message.additional_kwargs
                if isinstance(additional_kwargs, dict):
                    kw_calls = additional_kwargs.get("tool_calls")
                    if isinstance(kw_calls, list) and idx < len(kw_calls):
                        kw_calls[idx] = call

            message.tool_calls = tool_calls
