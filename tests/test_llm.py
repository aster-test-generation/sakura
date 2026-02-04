from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from sakura.utils.llm import LLMClient
from sakura.utils.llm.model import ClientType


class MockAPIError(Exception):
    """Mock exception with status_code attribute for testing."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class TestLLMClientRetry:
    """Tests for LLMClient retry behavior using mocks."""

    @patch("sakura.utils.llm.llm_client.ChatOpenAI")
    @patch("sakura.utils.llm.llm_client.Config")
    def test_retry_on_rate_limit_succeeds_after_retry(
        self, mock_config_class, mock_chat_openai
    ):
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda section, key: {
            ("llm", "provider"): "openrouter",
            ("llm", "model"): "test-model",
            ("llm", "summarization_temp"): 0.7,
            ("llm", "api_url"): "https://api.test.com",
            ("llm", "api_key"): "test-key",
            ("llm", "max_tokens"): 1000,
            ("llm", "timeout"): 30,
            ("llm", "default_headers"): None,
            ("llm", "model_kwargs"): {},
            ("llm", "can_parallel_tool"): False,
        }.get((section, key))
        mock_config_class.return_value = mock_config

        mock_chat = MagicMock()
        mock_chat.model_name = "test-model"
        mock_chat_openai.return_value = mock_chat

        rate_limit_error = MockAPIError("Rate limit exceeded", 429)
        success_response = AIMessage(content="Success!")

        call_count = 0

        def invoke_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise rate_limit_error
            return success_response

        mock_chat.invoke.side_effect = invoke_side_effect

        client = LLMClient(ClientType.SUMMARIZATION)
        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hello"),
        ]

        with patch.object(client, "_build_runnable", return_value=mock_chat):
            result = client.invoke_messages(messages)

        assert result.content == "Success!"
        assert call_count == 3

    @patch("sakura.utils.llm.llm_client.ChatOpenAI")
    @patch("sakura.utils.llm.llm_client.Config")
    def test_no_retry_on_client_error(self, mock_config_class, mock_chat_openai):
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda section, key: {
            ("llm", "provider"): "openrouter",
            ("llm", "model"): "test-model",
            ("llm", "summarization_temp"): 0.7,
            ("llm", "api_url"): "https://api.test.com",
            ("llm", "api_key"): "test-key",
            ("llm", "max_tokens"): 1000,
            ("llm", "timeout"): 30,
            ("llm", "default_headers"): None,
            ("llm", "model_kwargs"): {},
            ("llm", "can_parallel_tool"): False,
        }.get((section, key))
        mock_config_class.return_value = mock_config

        mock_chat = MagicMock()
        mock_chat.model_name = "test-model"
        mock_chat_openai.return_value = mock_chat

        bad_request_error = MockAPIError("Invalid request", 400)

        call_count = 0

        def invoke_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise bad_request_error

        mock_chat.invoke.side_effect = invoke_side_effect

        client = LLMClient(ClientType.SUMMARIZATION)
        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hello"),
        ]

        with patch.object(client, "_build_runnable", return_value=mock_chat):
            with pytest.raises(MockAPIError, match="Invalid request"):
                client.invoke_messages(messages)

        assert call_count == 1


class TestLLMClientIntegration:
    def test_llm_client_invoke_messages(self, petclinic_config):
        llm = LLMClient(ClientType.SUMMARIZATION)
        system = "You are a helpful assistant. Respond briefly."
        query = "Say hello in exactly one word."
        messages = [SystemMessage(content=system), HumanMessage(content=query)]

        result = llm.invoke_messages(messages)
        assert result is not None
        assert isinstance(result, AIMessage)
        assert result.content

    def test_llm_client_invoke_prompts(self, petclinic_config):
        llm = LLMClient(ClientType.SUMMARIZATION)
        system = "You are a helpful assistant. Respond briefly."
        chat = "What is 2 + 2? Answer with just the number."

        result = llm.invoke_prompts(system, chat)
        assert result is not None
        assert isinstance(result, AIMessage)
        assert result.content
