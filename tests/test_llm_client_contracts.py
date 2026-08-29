from __future__ import annotations

from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableSerializable
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from tenacity import wait_none

from sakura.utils.llm.llm_client import LLMClient
from sakura.utils.llm.model import ClientType
from sakura.utils.llm.usage_tracker import UsageTracker


class Answer(BaseModel):
    value: str


def _config_values(
    *,
    default_headers: dict[str, str] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[tuple[str, str], Any]:
    return {
        ("llm", "api_url"): "https://api.test/v1",
        ("llm", "provider"): "openrouter",
        ("llm", "model"): "test-model",
        ("llm", "summarization_temp"): 0.3,
        ("llm", "api_key"): "test-key",
        ("llm", "max_tokens"): 1000,
        ("llm", "timeout"): 30,
        ("llm", "default_headers"): default_headers,
        ("llm", "model_kwargs"): model_kwargs or {},
        ("llm", "can_parallel_tool"): False,
        ("reasoning", "configure"): True,
        ("reasoning", "effort"): "medium",
        ("reasoning", "exclude"): True,
        ("openrouter", "ignore_providers"): ["ignored-provider"],
    }


def _make_client(
    *,
    usage_tracker: UsageTracker | None = None,
) -> tuple[LLMClient, MagicMock]:
    config = MagicMock()
    values = _config_values()
    config.get.side_effect = lambda section, key: values[(section, key)]

    chat = MagicMock()
    chat.model_name = "test-model"
    with (
        patch("sakura.utils.llm.llm_client.Config", return_value=config),
        patch("sakura.utils.llm.llm_client.ChatOpenAI", return_value=chat),
    ):
        client = LLMClient(
            ClientType.SUMMARIZATION,
            usage_tracker=usage_tracker,
        )
    return client, chat


def test_metadata_less_response_still_counts_call_and_normalizes_tool_id() -> None:
    tracker = UsageTracker()
    client, _ = _make_client(usage_tracker=tracker)
    runnable = MagicMock(spec=RunnableSerializable)
    runnable.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{"name": "lookup", "args": {}, "id": None}],
    )

    with patch.object(client, "_build_runnable", return_value=runnable):
        result = client.invoke_messages([HumanMessage(content="Find it")])

    assert isinstance(result, AIMessage)
    tool_call_id = result.tool_calls[0]["id"]
    assert tool_call_id is not None
    assert tool_call_id.startswith("tool_")
    assert tracker.totals() == {
        "calls": 1,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def test_structured_response_returns_parsed_value_and_tracks_raw_usage() -> None:
    tracker = UsageTracker()
    client, _ = _make_client(usage_tracker=tracker)
    parsed = Answer(value="done")
    raw = AIMessage(
        content='{"value":"done"}',
        usage_metadata={
            "input_tokens": 7,
            "output_tokens": 3,
            "total_tokens": 10,
            "output_token_details": {"reasoning_tokens": 2},
        },
    )
    runnable = MagicMock(spec=RunnableSerializable)
    runnable.invoke.return_value = {
        "raw": raw,
        "parsed": parsed,
        "parsing_error": None,
    }

    with patch.object(client, "_build_runnable", return_value=runnable):
        result = client.invoke_messages(
            [HumanMessage(content="Answer")],
            schema=Answer,
        )

    assert result == parsed
    assert tracker.totals() == {
        "calls": 1,
        "input_tokens": 7,
        "output_tokens": 5,
    }


def test_structured_response_propagates_parsing_error_after_tracking_usage() -> None:
    tracker = UsageTracker()
    client, _ = _make_client(usage_tracker=tracker)
    parsing_error = ValueError("invalid structured response")
    raw = AIMessage(
        content="not-json",
        usage_metadata={
            "input_tokens": 4,
            "output_tokens": 1,
            "total_tokens": 5,
        },
    )
    runnable = MagicMock(spec=RunnableSerializable)
    runnable.invoke.return_value = {
        "raw": raw,
        "parsed": None,
        "parsing_error": parsing_error,
    }

    with (
        patch.object(client, "_build_runnable", return_value=runnable),
        pytest.raises(ValueError, match="invalid structured response") as exc_info,
    ):
        client.invoke_messages([HumanMessage(content="Answer")], schema=Answer)

    assert exc_info.value is parsing_error
    assert runnable.invoke.call_count == 1
    assert tracker.totals() == {
        "calls": 1,
        "input_tokens": 4,
        "output_tokens": 1,
    }


def test_transient_structured_parse_failure_is_retried() -> None:
    tracker = UsageTracker()
    client, _ = _make_client(usage_tracker=tracker)
    parsed = Answer(value="done")
    transient_error = ValueError(
        "Structured Output response does not have a 'parsed' field nor a "
        "'refusal' field."
    )
    failure = {
        "raw": AIMessage(
            content="",
            usage_metadata={
                "input_tokens": 5,
                "output_tokens": 0,
                "total_tokens": 5,
            },
        ),
        "parsed": None,
        "parsing_error": transient_error,
    }
    success = {
        "raw": AIMessage(
            content='{"value":"done"}',
            usage_metadata={
                "input_tokens": 5,
                "output_tokens": 2,
                "total_tokens": 7,
            },
        ),
        "parsed": parsed,
        "parsing_error": None,
    }
    runnable = MagicMock(spec=RunnableSerializable)
    runnable.invoke.side_effect = [failure, success]

    with (
        patch.object(client, "_build_runnable", return_value=runnable),
        patch.object(
            getattr(LLMClient._invoke_with_retry, "retry"), "wait", wait_none()
        ),
    ):
        result = client.invoke_messages(
            [HumanMessage(content="Answer")],
            schema=Answer,
        )

    assert result == parsed
    assert runnable.invoke.call_count == 2
    assert tracker.totals() == {
        "calls": 2,
        "input_tokens": 10,
        "output_tokens": 2,
    }


def test_tool_binding_receives_all_per_call_model_options() -> None:
    client, chat = _make_client()
    tool = MagicMock(spec=BaseTool)
    bound = MagicMock(spec=RunnableSerializable)
    chat.bind_tools.return_value = bound
    response_format = {"type": "json_object"}

    result = client._build_runnable(
        tools=[tool],
        tool_choice="required",
        response_format=response_format,
        extra_model_kwargs={"parallel_tool_calls": False, "seed": 17},
        temperature=0.1,
    )

    assert result is bound
    chat.bind_tools.assert_called_once_with(
        [tool],
        tool_choice="required",
        temperature=0.1,
        response_format=response_format,
        parallel_tool_calls=False,
        seed=17,
    )
    chat.bind.assert_not_called()


def test_structured_binding_receives_model_options_and_includes_raw() -> None:
    client, chat = _make_client()
    bound = MagicMock(spec=RunnableSerializable)
    chat.with_structured_output.return_value = bound

    result = client._build_runnable(
        schema=Answer,
        strict=False,
        method="function_calling",
        extra_model_kwargs={"seed": 23},
        temperature=0.2,
    )

    assert result is bound
    chat.with_structured_output.assert_called_once_with(
        schema=Answer,
        strict=False,
        method="function_calling",
        include_raw=True,
        temperature=0.2,
        seed=23,
    )
    chat.bind.assert_not_called()


def test_tools_and_schema_are_rejected_explicitly() -> None:
    client, chat = _make_client()

    with pytest.raises(
        ValueError,
        match="Tools and structured output cannot be combined",
    ):
        client._build_runnable(
            tools=[MagicMock(spec=BaseTool)],
            schema=Answer,
        )

    chat.bind_tools.assert_not_called()
    chat.with_structured_output.assert_not_called()


def test_client_construction_does_not_mutate_shared_config_values() -> None:
    default_headers = {"Existing": "header"}
    model_kwargs: dict[str, Any] = {
        "custom": {"nested": ["original"]},
        "extra_body": {"existing": {"enabled": True}},
    }
    original_headers = deepcopy(default_headers)
    original_model_kwargs = deepcopy(model_kwargs)
    values = _config_values(
        default_headers=default_headers,
        model_kwargs=model_kwargs,
    )
    config = MagicMock()
    config.get.side_effect = lambda section, key: values[(section, key)]
    chats = [MagicMock(model_name="test-model") for _ in range(2)]

    with (
        patch("sakura.utils.llm.llm_client.Config", return_value=config),
        patch(
            "sakura.utils.llm.llm_client.ChatOpenAI",
            side_effect=chats,
        ) as chat_class,
    ):
        LLMClient(ClientType.SUMMARIZATION)
        LLMClient(ClientType.SUMMARIZATION)

    assert default_headers == original_headers
    assert model_kwargs == original_model_kwargs
    assert len(chat_class.call_args_list) == 2

    first_kwargs = chat_class.call_args_list[0].kwargs
    second_kwargs = chat_class.call_args_list[1].kwargs
    assert first_kwargs == second_kwargs
    assert first_kwargs["default_headers"] == {
        "Existing": "header",
        "HTTP-Referer": "http://localhost",
        "X-Title": "NL2Test LLM Client",
    }
    assert first_kwargs["model_kwargs"] == {
        "custom": {"nested": ["original"]},
        "parallel_tool_calls": False,
    }
    assert first_kwargs["extra_body"] == {
        "existing": {"enabled": True},
        "reasoning": {"effort": "medium", "exclude": True},
        "provider": {"ignore": ["ignored-provider"]},
    }
    assert first_kwargs["default_headers"] is not default_headers
    assert first_kwargs["model_kwargs"] is not model_kwargs
    assert first_kwargs["extra_body"] is not model_kwargs["extra_body"]
