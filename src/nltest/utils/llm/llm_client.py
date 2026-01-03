from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, Optional, Sequence, Tuple, Union, Literal
import textwrap

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSerializable

from ..config.config import Config
from .model import Provider, LLMSettings, ClientType
from ..exceptions import ConfigurationException
from nltest.utils.pretty.color_logger import RichLog
import traceback
from .usage_tracker import UsageTracker


class LLMClient:
    def __init__(
        self,
        client_type: ClientType,
        *,
        usage_tracker: UsageTracker | None = None,
    ):
        config = Config()

        provider = config.get("llm", "provider")
        try:
            provider = Provider(provider)
        except ValueError:
            raise ConfigurationException(
                f"Invalid LLM provider: {provider}. Must be one of {list(Provider)}"
            )

        model = config.get("llm", "model")
        temp = config.get("llm", f"{client_type.value}_temp")

        base_url = config.get("llm", "api_url")
        api_key = config.get("llm", "api_key")

        if api_key is None:
            raise ConfigurationException(
                "API key for LLM provider is not set in the configuration."
            )

        # Assign default values if not set in config
        try:
            max_tokens = config.get("llm", "max_tokens")
        except ConfigurationException:
            max_tokens = 20000

        try:
            timeout = config.get("llm", "timeout")
        except ConfigurationException:
            timeout = None

        try:
            default_headers = config.get("llm", "default_headers")
        except ConfigurationException:
            default_headers = None

        try:
            model_kwargs = config.get("llm", "model_kwargs")
        except ConfigurationException:
            model_kwargs = {}

        if provider == Provider.OPENROUTER:
            default_headers = {} if default_headers is None else default_headers
            default_headers.setdefault("HTTP-Referer", "http://localhost")
            default_headers.setdefault("X-Title", "NL2Test LLM Client")

        # Configure parallel tool call behavior from Config
        try:
            can_parallel_tool = config.get("llm", "can_parallel_tool")
        except ConfigurationException:
            can_parallel_tool = False

        # Only set if not explicitly provided in model_kwargs.
        if "parallel_tool_calls" not in model_kwargs:
            model_kwargs["parallel_tool_calls"] = bool(can_parallel_tool)

        self._chat = ChatOpenAI(
            model=model,
            temperature=temp,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            default_headers=default_headers,
            model_kwargs=model_kwargs,
        )

        # Store model id for capability queries
        try:
            self._model_id = self._chat.model_name
        except Exception:
            self._model_id = model
        self._usage_tracker = usage_tracker or UsageTracker()
        # Save context for error logging
        self._provider = provider
        self._model = model
        self._base_url = base_url
        self._client_type = client_type

    def _build_runnable(
        self,
        *,
        tools: Optional[Sequence[BaseTool]] = None,
        tool_choice: Union[str, dict, None] = "auto",
        response_format: Optional[Dict[str, Any]] = None,
        extra_model_kwargs: Optional[Dict[str, Any]] = None,
        schema: Any = None,
        strict: bool = True,
        method: Optional[
            Literal["json_schema", "function_calling", "json_mode"]
        ] = "json_schema",
        temperature: Optional[float] = None,
    ) -> RunnableSerializable:
        runnable: RunnableSerializable = self._chat

        if temperature is not None:
            runnable = runnable.bind(temperature=temperature)

        if tools:
            runnable = runnable.bind_tools(tools, tool_choice=tool_choice)

        if (
            response_format is not None and schema is None
        ):  # NOTE: If schema is provided, we don't need to bind the response format
            runnable = runnable.bind(response_format=response_format)

        if extra_model_kwargs:
            runnable = runnable.bind(**extra_model_kwargs)

        if schema is not None:
            runnable = runnable.with_structured_output(
                schema=schema, strict=strict, method=method
            )

        return runnable

    @property
    def chat(self) -> ChatOpenAI:
        return self._chat

    def _normalize_tool_call_ids(self, ai_msg: AIMessage) -> AIMessage:
        """Ensure tool call IDs are always present by generating unique IDs if missing."""
        if getattr(ai_msg, "tool_calls", None):
            normalized = []
            for tc in ai_msg.tool_calls:
                tc_id = tc.get("id") or f"tool_{uuid.uuid4().hex[:8]}"
                tc["id"] = tc_id
                normalized.append(tc)
            ai_msg.tool_calls = normalized
        return ai_msg

    def invoke_messages(
        self,
        messages: Sequence[BaseMessage],
        *,
        tools: Optional[Sequence[BaseTool]] = None,
        tool_choice: Union[str, dict, None] = "auto",
        response_format: Optional[Dict[str, Any]] = None,
        extra_model_kwargs: Optional[Dict[str, Any]] = None,
        schema: Any = None,
        strict: bool = True,
        method: Optional[
            Literal["json_schema", "function_calling", "json_mode"]
        ] = "json_schema",
        temperature: Optional[float] = None,
    ) -> Any:
        runnable = self._build_runnable(
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            extra_model_kwargs=extra_model_kwargs,
            schema=schema,
            strict=strict,
            method=method,
            temperature=temperature,
        )
        try:
            out = runnable.invoke(list(messages))
        except Exception as e:
            err_type = type(e).__name__
            RichLog.error(
                f"[LLMClient] {err_type} during invoke (provider={getattr(self._provider, 'value', self._provider)}, "
                f"model={self._model}, base_url={self._base_url}, client={self._client_type.value}): {e}"
            )
            # Traceback helps pinpoint issues inside LangChain/OpenAI stack.
            RichLog.debug(traceback.format_exc())

            # Log options used for this call (debug only to avoid noise).
            RichLog.debug(
                f"opts: tool_choice={tool_choice}, "
                f"schema={'yes' if schema is not None else 'no'}, "
                f"response_format={'yes' if response_format is not None else 'no'}, "
                f"extra_model_kwargs={str(extra_model_kwargs)[:500]}"
            )
            # Summarize message types for quick inspection
            msg_types = [type(m).__name__ for m in messages]
            RichLog.debug(f"messages: {','.join(msg_types)}")

            raise

        # Record usage if we have token information
        if hasattr(out, "usage_metadata") and out.usage_metadata:
            self._usage_tracker.record(
                input_tokens=out.usage_metadata.get("input_tokens", 0),
                output_tokens=out.usage_metadata.get("output_tokens", 0),
            )

        return self._normalize_tool_call_ids(out) if isinstance(out, AIMessage) else out

    def invoke_prompts(
        self,
        system: str,
        chat: str,
        *,
        response_format: Optional[Dict[str, Any]] = None,
        extra_model_kwargs: Optional[Dict[str, Any]] = None,
        schema: Any = None,
        strict: bool = True,
        method: Optional[
            Literal["json_schema", "function_calling", "json_mode"]
        ] = "json_schema",
        temperature: Optional[float] = None,
    ) -> Any:
        messages: Sequence[BaseMessage] = [
            SystemMessage(content=system),
            HumanMessage(content=chat),
        ]
        return self.invoke_messages(
            messages,
            response_format=response_format,
            extra_model_kwargs=extra_model_kwargs,
            schema=schema,
            strict=strict,
            method=method,
            temperature=temperature,
        )

    @staticmethod
    def sanitize(text: str) -> str:
        # Apply standard sanitation operations
        sanitize_operations = [
            lambda t: re.sub(r"(?si).*?</think>", "", t, flags=re.IGNORECASE),
            # Remove everything before and include </think>
        ]
        for op in sanitize_operations:
            text = op(text)
        return text.strip()

    @staticmethod
    def _normalize_string_values(value: Any) -> Any:
        """Recursively dedent and strip string values to normalize LLM tool args."""
        if isinstance(value, str):
            return textwrap.dedent(value).strip()
        if isinstance(value, dict):
            return {k: LLMClient._normalize_string_values(v) for k, v in value.items()}
        if isinstance(value, list):
            return [LLMClient._normalize_string_values(v) for v in value]
        if isinstance(value, tuple):
            return tuple(LLMClient._normalize_string_values(v) for v in value)
        return value

    @staticmethod
    def parse_tool_args(args: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Tool call args can be passed as a JSON string or a dict."""
        if isinstance(args, dict):
            return LLMClient._normalize_string_values(args)
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
            except Exception:
                return {}
            if not isinstance(parsed, dict):
                return {}
            return LLMClient._normalize_string_values(parsed)
        return {}
