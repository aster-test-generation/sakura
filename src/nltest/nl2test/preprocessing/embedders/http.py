from typing import List

import requests

from .base import BaseEmbedder
from nltest.utils.config import Config
from nltest.utils.pretty.color_logger import RichLog


class HttpEmbedder(BaseEmbedder):
    def __init__(self, model_id: str, api_url: str):
        """HTTP embedder that calls an OpenAI-compatible embeddings endpoint."""
        self.model_id = model_id
        self.api_url = self._normalize_api_url(api_url)

        # Retrieve API key from config
        try:
            self.api_key = Config().get("emb", "api_key")
        except Exception:
            self.api_key = None

        # Probe to determine vector dimension once for downstream vector stores.
        probe_embedding = self._embed("probe")
        dim = len(probe_embedding)
        super().__init__(dim)

    @staticmethod
    def _normalize_api_url(api_url: str) -> str:
        """Always append '/embeddings' to the provided API URL."""
        return f"{(api_url or '').rstrip('/')}/embeddings"

    def _embed(self, text: str) -> List[float]:
        if not text:
            raise ValueError("Text to embed cannot be empty")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "input": text,
            "model": self.model_id,
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            http_data = result.get("data", {})

            if not isinstance(http_data, list) or not http_data:
                # Log invalid structure for debugging then raise
                RichLog.error(
                    f"[HttpEmbedder] Invalid embedding response structure (model={self.model_id})."
                )
                RichLog.debug(f"raw json: {str(result)[:1000]}")
                raise ValueError("Invalid embedding response from API")

            embedding = http_data[0].get("embedding")

            if not isinstance(embedding, list) or not embedding:
                RichLog.error(
                    f"[HttpEmbedder] Missing embedding field in response (model={self.model_id})."
                )
                RichLog.debug(f"raw json: {str(result)[:1000]}")
                raise ValueError("Invalid embedding response from API")

            return embedding

        except requests.RequestException as e:
            status = getattr(e.response, "status_code", None) if getattr(e, "response", None) else None
            text_snip = (
                (e.response.text[:1000] if e.response and isinstance(e.response.text, str) else "")
            )
            RichLog.error(
                f"[HttpEmbedder] Request failed (status={status}) model={self.model_id} url={self.api_url}: {e}"
            )
            if text_snip:
                RichLog.debug(f"response body: {text_snip}")
            raise RuntimeError(f"HTTP embedding request failed: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)
