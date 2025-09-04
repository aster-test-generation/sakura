from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, Optional, Sequence, Tuple, Union, Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSerializable

from ..config.config import Config
from .model import Provider, LLMSettings, ClientType
from ..constants import PARALLEL_TOOL_CALLABLE
from ..exceptions import ConfigurationException
from .usage_tracker import usage_tracker


class LLMClient:
    def __init__(self, client_type: ClientType):
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
            max_tokens = 10000

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
            # Defer parallel tool call enablement to can_parallel_tool_call()
            if "parallel_tool_calls" not in model_kwargs:
                # Set a sensible default based on known model capabilities
                model_kwargs["parallel_tool_calls"] = (
                    model in PARALLEL_TOOL_CALLABLE.get(True, set())
                )

            # if "reasoning" not in model_kwargs:
            #    model_kwargs["reasoning"] = {"enabled": True}
            #    model_kwargs["reasoning"]["effort"] = "medium"

            # if "include_reasoning" not in model_kwargs:
            #    model_kwargs["include_reasoning"] = True

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
    ) -> RunnableSerializable:
        runnable: RunnableSerializable = self._chat

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

    def can_parallel_tool_call(self) -> bool:
        """
        Return True if the current model supports parallel tool calls.
        """
        model_id = getattr(self, "_model_id", None) or self._chat.model_name
        if model_id in PARALLEL_TOOL_CALLABLE.get(True, set()):
            return True
        if model_id in PARALLEL_TOOL_CALLABLE.get(False, set()):
            return False
        return False

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
    ) -> Any:
        runnable = self._build_runnable(
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            extra_model_kwargs=extra_model_kwargs,
            schema=schema,
            strict=strict,
            method=method,
        )
        out = runnable.invoke(list(messages))

        # Record usage if we have token information
        if hasattr(out, "usage_metadata") and out.usage_metadata:
            usage_tracker.record(
                model=self._chat.model_name,
                input_tokens=out.usage_metadata.get("input_tokens", 0),
                output_tokens=out.usage_metadata.get("output_tokens", 0),
                total_tokens=out.usage_metadata.get("total_tokens", 0),
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
    def parse_tool_args(args: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Tool call args can be passed as a JSON string or a dict."""
        if isinstance(args, dict):
            return args
        try:
            return json.loads(args)
        except Exception:
            return {}
