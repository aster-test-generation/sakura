from typing import List
from urllib.parse import urlparse, urlunparse

import requests

from .base import BaseEmbedder
from nltest.utils.config import Config


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
        """Append '/embeddings' when given a base '/v1' URL; else leave as-is."""
        try:
            parsed = urlparse(api_url)
            path = (parsed.path or "").rstrip("/")
            is_v1_base = path.endswith("/v1")
            already_embeddings = path.endswith("/embeddings")
            if is_v1_base and not already_embeddings:
                new_path = f"{path}/embeddings"
                parsed = parsed._replace(path=new_path)
                return urlunparse(parsed)
            return api_url
        except Exception:
            return api_url

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
                raise ValueError("Invalid embedding response from API")
            embedding = http_data[0].get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise ValueError("Invalid embedding response from API")
            return embedding
        except requests.RequestException as e:
            raise RuntimeError(f"HTTP embedding request failed: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)
