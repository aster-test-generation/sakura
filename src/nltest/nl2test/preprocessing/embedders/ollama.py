from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import ollama
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception,
    RetryCallState,
)

from .base import BaseEmbedder
from nltest.utils.pretty.color_logger import RichLog

OLLAMA_MAX_WORKERS = 4


def _is_retriable_error(exc: BaseException) -> bool:
    """Check if exception is transient and worth retrying."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    msg = str(exc).lower()
    if "rate limit" in msg or "too many requests" in msg or "overloaded" in msg:
        return True
    if "connection" in msg or "timeout" in msg:
        return True
    return False


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log retry attempts for observability."""
    attempt = retry_state.attempt_number
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    wait = retry_state.next_action.sleep if retry_state.next_action else 0
    RichLog.warn(
        f"[OllamaEmbedder] Retry attempt {attempt} after {wait:.1f}s due to: "
        f"{type(exc).__name__ if exc else 'unknown'}"
    )


class OllamaEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_id: str = "dengcao/Qwen3-Embedding-4B:Q5_K_M",
        dim: Optional[int] = None,
    ):
        self.model_id = model_id

        if dim is None:
            probe = self._raw_embed("probe")
            dim = int(probe.shape[0])
        super().__init__(dim)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=30),
        retry=retry_if_exception(_is_retriable_error),
        before_sleep=_log_retry_attempt,
        reraise=True,
    )
    def _raw_embed(self, text: str) -> np.ndarray:
        if not text:
            raise ValueError("Text to embed cannot be empty")
        try:
            raw = ollama.embeddings(model=self.model_id, prompt=text)["embedding"]
        except Exception as e:
            raise RuntimeError(f"Error embedding with Ollama: {e}")
        return np.asarray(raw, dtype=np.float32).reshape(-1)

    def _embed(self, text: str) -> List[float]:
        arr = self._raw_embed(text)
        if arr.shape[0] != self.dim:
            raise ValueError(
                f"Ollama returned a vector of length {arr.shape[0]}, expected {self.dim}"
            )
        return arr.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        with ThreadPoolExecutor(max_workers=OLLAMA_MAX_WORKERS) as executor:
            batch = list(executor.map(self._embed, texts))
        arr = np.asarray(batch, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(
                f"Batch embeddings have invalid shape {arr.shape}, expected (N, {self.dim})"
            )
        return arr.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)
