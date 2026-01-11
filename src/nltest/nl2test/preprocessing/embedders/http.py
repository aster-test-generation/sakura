from typing import List, Optional

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception,
    RetryCallState,
)

from .base import BaseEmbedder
from nltest.utils.pretty.color_logger import RichLog

EMBED_CHUNK_SIZE = 100


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
        f"[HttpEmbedder] Retry attempt {attempt} after {wait:.1f}s due to: "
        f"{type(exc).__name__ if exc else 'unknown'}"
    )


class HttpEmbedder(BaseEmbedder):
    def __init__(self, model_id: str, api_url: str, api_key: Optional[str] = None):
        # Note: max_retries is omitted to let tenacity handle all retry logic
        # with proper exponential backoff. Add max_retries here if you want
        # LangChain's built-in HTTP-level retries to stack with tenacity.
        self._client = OpenAIEmbeddings(
            model=model_id,
            base_url=api_url.rstrip("/"),
            api_key=SecretStr(api_key) if api_key else None,
            check_embedding_ctx_length=False,
        )
        probe_embedding = self._embed_query_with_retry("probe")
        super().__init__(len(probe_embedding))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=30),
        retry=retry_if_exception(_is_retriable_error),
        before_sleep=_log_retry_attempt,
        reraise=True,
    )
    def _embed_chunk_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Embed a chunk of texts with retry logic."""
        return self._client.embed_documents(texts)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=30),
        retry=retry_if_exception(_is_retriable_error),
        before_sleep=_log_retry_attempt,
        reraise=True,
    )
    def _embed_query_with_retry(self, text: str) -> List[float]:
        """Embed a single query with retry logic."""
        return self._client.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if len(texts) <= EMBED_CHUNK_SIZE:
            return self._embed_chunk_with_retry(texts)
        results: List[List[float]] = []
        for i in range(0, len(texts), EMBED_CHUNK_SIZE):
            chunk = texts[i : i + EMBED_CHUNK_SIZE]
            results.extend(self._embed_chunk_with_retry(chunk))
        return results

    def embed_query(self, text: str) -> List[float]:
        return self._embed_query_with_retry(text)
