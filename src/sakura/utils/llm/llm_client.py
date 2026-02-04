from __future__ import annotations

import json
import re
import textwrap
import traceback
import uuid
from typing import Any, Dict, Literal, Optional, Sequence, Type, TypeVar, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableSerializable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from openai import LengthFinishReasonError
from pydantic import BaseModel, SecretStr
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_chain,
    wait_random,
)

from sakura.utils.pretty.color_logger import RichLog

from ..config.config import Config
from ..exceptions import ConfigurationException
from .model import ClientType, Provider
from .usage_tracker import UsageTracker

from .general_prompts.structured_retry import (
    LENGTH_EXCEEDED_RETRY_PROMPT,
    STRUCTURED_OUTPUT_RETRY_PROMPT,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class EmptyLLMResponseError(RuntimeError):
    """Raised when the provider returns an AIMessage with no content and no tool calls."""


def _is_retriable_error(exc: BaseException) -> bool:
    """Check if exception is transient and worth retrying."""
    if isinstance(exc, EmptyLLMResponseError):
        return True
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
    # Handle empty structured output responses (common with some models via OpenRouter)
    if isinstance(
        exc, ValueError
    ) and "does not have a 'parsed' field nor a 'refusal' field" in str(exc):
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
            max_tokens = 16384

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

        # Extract extra_body from model_kwargs if present (should be a direct kwarg)
        extra_body: dict[str, Any] = model_kwargs.pop("extra_body", {})

        # The reasoning parameter is OpenRouter-specific and should only be sent
        # when explicitly configured (configure_reasoning=True)
        try:
            configure_reasoning = config.get("reasoning", "configure")
        except ConfigurationException:
            configure_reasoning = False

        if provider == Provider.OPENROUTER and configure_reasoning:
            try:
                reasoning_effort = config.get("reasoning", "effort")
            except ConfigurationException:
                reasoning_effort = "low"
            try:
                reasoning_exclude = config.get("reasoning", "exclude")
            except ConfigurationException:
                reasoning_exclude = False
            extra_body["reasoning"] = {
                "effort": reasoning_effort,
                "exclude": reasoning_exclude,
            }

        if provider == Provider.OPENROUTER:
            default_headers = {} if default_headers is None else default_headers
            default_headers.setdefault("HTTP-Referer", "http://localhost")
            default_headers.setdefault("X-Title", "NL2Test LLM Client")

            try:
                ignore_providers = config.get("openrouter", "ignore_providers")
                if ignore_providers:
                    if "provider" not in extra_body:
                        extra_body["provider"] = {}
                    extra_body["provider"]["ignore"] = ignore_providers
            except ConfigurationException:
                pass

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
        # Mistral API requires max_tokens in extra_body (rejects max_completion_tokens)
        if provider == Provider.MISTRAL:
            extra_body["max_tokens"] = max_tokens
            max_tokens_kwarg = {}
        else:
            max_tokens_kwarg = {"max_tokens": max_tokens}
        self._chat = ChatOpenAI(
            model=model,
            temperature=temp,
            **max_tokens_kwarg,
            base_url=base_url.rstrip("/") if base_url else None,
            api_key=SecretStr(api_key),
            timeout=timeout,
            default_headers=default_headers,
            model_kwargs=model_kwargs,
            extra_body=extra_body if extra_body else None,
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

    @staticmethod
    def _message_preview(message: BaseMessage | None, limit: int = 500) -> str:
        if message is None:
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            segments = []
            for chunk in content:
                if isinstance(chunk, dict):
                    segments.append(
                        str(chunk.get("text") or chunk.get("content") or chunk)
                    )
                else:
                    segments.append(str(chunk))
            text = "\n".join(segments)
        else:
            text = str(content)
        text = text.strip()
        return text[:limit]

    def _summarize_messages(self, messages: Sequence[BaseMessage]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "count": len(messages),
            "types": {},
        }
        for msg in messages:
            msg_type = type(msg).__name__
            summary["types"][msg_type] = summary["types"].get(msg_type, 0) + 1

        last_system = next(
            (m for m in reversed(messages) if isinstance(m, SystemMessage)), None
        )
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
        )
        last_tool = next(
            (m for m in reversed(messages) if isinstance(m, ToolMessage)), None
        )
        last_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage)), None
        )

        summary["last_message_type"] = type(messages[-1]).__name__ if messages else None
        summary["system_preview"] = self._message_preview(last_system)
        summary["last_human_preview"] = self._message_preview(last_human)
        summary["last_tool_preview"] = self._message_preview(last_tool)
        summary["last_ai_preview"] = self._message_preview(last_ai)

        if last_ai is not None:
            tool_calls = getattr(last_ai, "tool_calls", None) or []
            summary["last_ai_tool_calls"] = [
                tc.get("name") for tc in tool_calls if isinstance(tc, dict)
            ]

        return summary

    def _log_empty_response(
        self,
        out: AIMessage,
        messages: Sequence[BaseMessage],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        def _content_len(value: Any) -> int:
            if isinstance(value, str):
                return len(value)
            if isinstance(value, list):
                return len(value)
            if value is None:
                return 0
            return len(str(value))

        payload = {
            "provider": getattr(self._provider, "value", self._provider),
            "model": self._model,
            "base_url": self._base_url,
            "client": self._client_type.value,
            "context": context or {},
            "response": {
                "content_type": type(out.content).__name__,
                "content_len": _content_len(out.content),
                "tool_calls": getattr(out, "tool_calls", None),
                "usage_metadata": getattr(out, "usage_metadata", None),
                "response_metadata": getattr(out, "response_metadata", None),
            },
            "messages": self._summarize_messages(messages),
        }

        RichLog.error(
            f"[LLMClient] empty_response details={json.dumps(payload, default=str)[:4000]}"
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_chain(
            wait_random(1, 2),
            wait_random(5, 10),
            wait_random(20, 30),
            wait_random(45, 60),
        ),
        retry=retry_if_exception(_is_retriable_error),
        before_sleep=_log_retry_attempt,
        reraise=True,
    )
    def _invoke_with_retry(
        self,
        runnable: RunnableSerializable,
        messages: Sequence[BaseMessage],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Internal method that performs the actual invocation with retry logic."""
        out = runnable.invoke(list(messages))

        if isinstance(out, AIMessage):
            # Treat "successful but empty" LLM responses as errors (likely provider problem) and retry.
            has_tool_calls = bool(getattr(out, "tool_calls", None))
            if isinstance(out.content, str):
                has_content = bool(out.content.strip())
            elif isinstance(out.content, list):
                has_content = len(out.content) > 0
            else:
                has_content = bool(out.content)
            if not has_tool_calls and not has_content:
                self._log_empty_response(out, messages, context)
                raise EmptyLLMResponseError(
                    "LLM returned an empty AIMessage (no content, no tool_calls)."
                )

        return out

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
        context: Optional[Dict[str, Any]] = None,
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
            out = self._invoke_with_retry(runnable, messages, context)
        except Exception as e:
            err_type = type(e).__name__
            RichLog.error(
                f"[LLMClient] {err_type} during invoke (provider={getattr(self._provider, 'value', self._provider)}, "
                f"model={self._model}, base_url={self._base_url}, client={self._client_type.value}): {e}"
            )
            RichLog.debug(traceback.format_exc())
            if context:
                RichLog.debug(f"context: {json.dumps(context, default=str)[:2000]}")
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
            output_tokens = out.usage_metadata.get("output_tokens", 0)
            # Include reasoning tokens in output count (billed as output tokens)
            output_details = out.usage_metadata.get("output_token_details") or {}
            reasoning_tokens = output_details.get("reasoning_tokens", 0)
            self._usage_tracker.record(
                input_tokens=out.usage_metadata.get("input_tokens", 0),
                output_tokens=output_tokens + reasoning_tokens,
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

    def invoke_structured_with_retries(
        self,
        *,
        messages: Sequence[BaseMessage] | None = None,
        system: str | None = None,
        chat: str | None = None,
        schema: Type[SchemaT],
        strict: bool = True,
        max_attempts: int = 3,
        retry_prompt_template: str | None = None,
        on_failure: Literal["raise", "return_none"] = "raise",
    ) -> SchemaT | None:
        """
        Invoke LLM with structured output binding, retrying on validation failures.

        Accepts EITHER:
        - messages: A pre-built message list (must end with HumanMessage)
        - system + chat: Simple prompt pair (for decomposer use cases)

        On failure, modifies the last HumanMessage to include retry context.
        """
        if messages is not None and (system is not None or chat is not None):
            raise ValueError("Provide either 'messages' OR 'system'+'chat', not both.")
        if messages is None and (system is None or chat is None):
            raise ValueError(
                "Must provide either 'messages' or both 'system' and 'chat'."
            )

        template = retry_prompt_template or STRUCTURED_OUTPUT_RETRY_PROMPT
        use_message_mode = messages is not None

        base_messages: list[BaseMessage]
        original_human_content: str
        if use_message_mode:
            base_messages = list(messages)
            if not base_messages or not isinstance(base_messages[-1], HumanMessage):
                raise ValueError(
                    "When using messages mode, the last message must be a HumanMessage."
                )
            last_human = base_messages[-1]
            original_human_content = (
                last_human.content
                if isinstance(last_human.content, str)
                else str(last_human.content)
            )
        else:
            assert system is not None and chat is not None
            base_messages = [SystemMessage(content=system)]
            original_human_content = chat

        failures: list[str] = []
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            working_messages = list(base_messages)

            if failures:
                failure_block = "\n\n".join(failures)
                augmented_content = f"{original_human_content}\n\n{failure_block}"
            else:
                augmented_content = original_human_content

            if use_message_mode:
                working_messages[-1] = HumanMessage(content=augmented_content)
            else:
                working_messages.append(HumanMessage(content=augmented_content))

            try:
                result = self.invoke_messages(
                    working_messages,
                    schema=schema,
                    strict=strict,
                )
                return result
            except LengthFinishReasonError as length_exc:
                last_error = length_exc
                # Extract partial output from the completion if available
                partial_content = ""
                completion = getattr(length_exc, "completion", None)
                if completion:
                    choices = getattr(completion, "choices", [])
                    if choices:
                        message = getattr(choices[0], "message", None)
                        if message:
                            partial_content = getattr(message, "content", "") or ""

                retry_prompt = LENGTH_EXCEEDED_RETRY_PROMPT.format(
                    attempt=attempt,
                    partial_output=partial_content[:500]
                    if partial_content
                    else "(none available)",
                )
                failures.append(retry_prompt)
            except Exception as exc:
                # Don't retry transient errors - tenacity already exhausted retries
                if _is_retriable_error(exc):
                    raise
                last_error = exc
                error_excerpt = (str(exc) or repr(exc)).strip()[:800]

                raw_output = getattr(exc, "raw_output", None)
                if raw_output is None:
                    response = getattr(exc, "response", None)
                    if response is not None:
                        raw_output = getattr(response, "text", None)
                output_excerpt = str(raw_output or "")[:800]

                retry_prompt = template.format(
                    attempt=attempt,
                    error_excerpt=error_excerpt,
                    output_excerpt=output_excerpt,
                )
                failures.append(retry_prompt)

        if on_failure == "return_none":
            return None

        if last_error is not None:
            raise last_error
        raise RuntimeError(
            "Structured prompt invocation failed without raising an error."
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
