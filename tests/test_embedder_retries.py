from unittest.mock import MagicMock, patch

import httpx
import ollama
import openai
import pytest
from tenacity import wait_none

from sakura.nl2test.preprocessing.embedders import http as http_embedder
from sakura.nl2test.preprocessing.embedders import ollama as ollama_embedder


class _StatusError(Exception):
    def __init__(self, status_code: int, message: str | None = None) -> None:
        super().__init__(message or f"status code: {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
def test_http_retry_predicate_accepts_transient_statuses(status_code: int) -> None:
    assert http_embedder._is_retriable_error(_StatusError(status_code))


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_http_retry_predicate_rejects_other_client_errors(status_code: int) -> None:
    assert not http_embedder._is_retriable_error(_StatusError(status_code))


def test_http_status_takes_precedence_over_retryable_message() -> None:
    error = _StatusError(400, "rate limit")

    assert not http_embedder._is_retriable_error(error)


def test_http_retry_predicate_accepts_openai_connection_errors() -> None:
    request = httpx.Request("POST", "https://embedding.invalid")

    assert http_embedder._is_retriable_error(openai.APIConnectionError(request=request))
    assert http_embedder._is_retriable_error(openai.APITimeoutError(request))


def test_http_embedder_disables_client_retries() -> None:
    client = MagicMock()
    client.embed_query.return_value = [0.1, 0.2]

    with patch.object(
        http_embedder, "OpenAIEmbeddings", return_value=client
    ) as factory:
        embedder = http_embedder.HttpEmbedder(
            model_id="test-model",
            api_url="https://embedding.invalid/",
            api_key="test-key",
        )

    assert embedder.dim == 2
    factory.assert_called_once()
    assert factory.call_args.kwargs["max_retries"] == 0


def test_http_embedder_retries_openai_connection_error_without_waiting() -> None:
    embedder = object.__new__(http_embedder.HttpEmbedder)
    client = MagicMock()
    request = httpx.Request("POST", "https://embedding.invalid")
    client.embed_query.side_effect = [
        openai.APIConnectionError(request=request),
        [0.1, 0.2],
    ]
    embedder._client = client

    with patch.object(
        getattr(http_embedder.HttpEmbedder._embed_query_with_retry, "retry"),
        "wait",
        wait_none(),
    ):
        result = embedder._embed_query_with_retry("query")

    assert result == [0.1, 0.2]
    assert client.embed_query.call_count == 2


@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
def test_ollama_retry_predicate_accepts_transient_statuses(status_code: int) -> None:
    error = ollama.ResponseError("server error", status_code)

    assert ollama_embedder._is_retriable_error(error)


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_ollama_retry_predicate_rejects_other_client_errors(status_code: int) -> None:
    error = ollama.ResponseError("client error", status_code)

    assert not ollama_embedder._is_retriable_error(error)


def test_ollama_status_takes_precedence_over_retryable_message() -> None:
    error = ollama.ResponseError("too many requests", 400)

    assert not ollama_embedder._is_retriable_error(error)


def test_ollama_embedder_retries_response_error_without_waiting() -> None:
    embedder = ollama_embedder.OllamaEmbedder(model_id="test-model", dim=2)
    first_error = ollama.ResponseError("server error", 503)

    with (
        patch.object(
            getattr(ollama_embedder.OllamaEmbedder._raw_embed, "retry"),
            "wait",
            wait_none(),
        ),
        patch.object(
            ollama_embedder.ollama,
            "embeddings",
            side_effect=[first_error, {"embedding": [0.1, 0.2]}],
        ) as embeddings,
    ):
        result = embedder._raw_embed("query")

    assert result.tolist() == pytest.approx([0.1, 0.2])
    assert embeddings.call_count == 2


def test_ollama_embedder_preserves_non_retriable_response_error() -> None:
    embedder = ollama_embedder.OllamaEmbedder(model_id="test-model", dim=2)
    error = ollama.ResponseError("bad request", 400)

    with patch.object(ollama_embedder.ollama, "embeddings", side_effect=error):
        with pytest.raises(ollama.ResponseError) as exc_info:
            embedder._raw_embed("query")

    assert exc_info.value is error


def test_ollama_embedder_chains_other_client_errors() -> None:
    embedder = ollama_embedder.OllamaEmbedder(model_id="test-model", dim=2)
    error = ValueError("malformed response")

    with patch.object(ollama_embedder.ollama, "embeddings", side_effect=error):
        with pytest.raises(RuntimeError) as exc_info:
            embedder._raw_embed("query")

    assert exc_info.value.__cause__ is error
