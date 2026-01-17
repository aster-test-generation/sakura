from __future__ import annotations

import json
import re
import textwrap
import traceback
import uuid
from typing import Any, Dict, Literal, Optional, Sequence, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableSerializable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from nltest.utils.pretty.color_logger import RichLog

from ..config.config import Config
from ..exceptions import ConfigurationException
from .model import ClientType, Provider
from .usage_tracker import UsageTracker


def _is_retriable_error(exc: BaseException) -> bool:
    """Check if exception is transient and worth retrying."""
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if status in {429, 500, 502, 503, 504}:
        return True
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    msg = str(exc).lower()
    if "rate limit" in msg or "too many requests" in msg or "overloaded" in msg:
        return True
    return False


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log retry attempts for observability."""
    attempt = retry_state.attempt_number
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    wait = retry_state.next_action.sleep if retry_state.next_action else 0
    RichLog.warn(
        f"[LLMClient] Retry attempt {attempt} after {wait:.1f}s due to: "
        f"{type(exc).__name__ if exc else 'unknown'}"
    )


class LLMClient:
    def __init__(
        self,
        client_type: ClientType,
        *,
        usage_tracker: UsageTracker | None = None,
    ):
        config = Config()

        base_url = config.get("llm", "api_url")
        provider_raw = config.get("llm", "provider")
        if provider_raw in (None, ""):
            provider = None
        else:
            try:
                provider = Provider(provider_raw)
            except ValueError as exc:
                raise ConfigurationException(
                    f"Invalid LLM provider: {provider_raw}. Must be one of {list(Provider)}"
                ) from exc
        if provider is None and not base_url:
            raise ConfigurationException(
                "LLM provider is not configured and no API URL was supplied."
            )

        model = config.get("llm", "model")
        temp = config.get("llm", f"{client_type.value}_temp")

        api_key = config.get("llm", "api_key")

        # Assign default values if not set in config
        try:
            max_tokens = config.get("llm", "max_tokens")
        except ConfigurationException:
            max_tokens = 24000

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

        try:
            reasoning_enabled = config.get("reasoning", "enabled")
        except ConfigurationException:
            reasoning_enabled = False

        if reasoning_enabled:
            try:
                reasoning_effort = config.get("reasoning", "effort")
            except ConfigurationException:
                reasoning_effort = "low"
            reasoning_obj = {"effort": reasoning_effort, "exclude": True}
            if "extra_body" not in model_kwargs:
                model_kwargs["extra_body"] = {}
            model_kwargs["extra_body"]["reasoning"] = reasoning_obj

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

        # Note: max_retries is omitted to let tenacity handle all retry logic
        # with proper exponential backoff. Add max_retries here if you want
        # LangChain's built-in HTTP-level retries to stack with tenacity.
        self._chat = ChatOpenAI(
            model=model,
            temperature=temp,
            max_tokens=max_tokens,
            base_url=base_url.rstrip("/") if base_url else None,
            api_key=SecretStr(api_key),
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

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=60),
        retry=retry_if_exception(_is_retriable_error),
        before_sleep=_log_retry_attempt,
        reraise=True,
    )
    def _invoke_with_retry(
        self, runnable: RunnableSerializable, messages: Sequence[BaseMessage]
    ) -> Any:
        """Internal method that performs the actual invocation with retry logic."""
        return runnable.invoke(list(messages))

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
            out = self._invoke_with_retry(runnable, messages)
        except Exception as e:
            err_type = type(e).__name__
            RichLog.error(
                f"[LLMClient] {err_type} during invoke (provider={getattr(self._provider, 'value', self._provider)}, "
                f"model={self._model}, base_url={self._base_url}, client={self._client_type.value}): {e}"
            )
            RichLog.debug(traceback.format_exc())
            RichLog.debug(
                f"opts: tool_choice={tool_choice}, "
                f"schema={'yes' if schema is not None else 'no'}, "
                f"response_format={'yes' if response_format is not None else 'no'}, "
                f"extra_model_kwargs={str(extra_model_kwargs)[:500]}"
            )
            msg_types = [type(m).__name__ for m in messages]
            RichLog.debug(f"messages: {','.join(msg_types)}")
            raise

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
