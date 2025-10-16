from typing import List
from urllib.parse import urlparse, urlunparse

import requests

from .base import BaseEmbedder


class HttpEmbedder(BaseEmbedder):
    def __init__(self, model_id: str, api_url: str):
        self.model_id = model_id
        # Normalize the API URL for OpenAI-compatible backends.
        # If the path ends with /v1 and does not already end with /embeddings,
        # append /embeddings so callers can pass a base URL like .../v1.
        self.api_url = self._normalize_api_url(api_url)

        # Probe for embedding dimension
        probe_embedding = self._embed("probe")
        dim = len(probe_embedding)
        super().__init__(dim)

    @staticmethod
    def _normalize_api_url(api_url: str) -> str:
        """Ensure embeddings endpoint when a base /v1 URL is provided."""
        try:
            parsed = urlparse(api_url)
            path = parsed.path or ""
            # Treat '/v1/' and '/v1' the same; only append if not already embeddings
            path_no_trailing = path.rstrip("/")
            is_v1_base = path_no_trailing.endswith("/v1")
            already_embeddings = path_no_trailing.endswith("/embeddings")
            if is_v1_base and not already_embeddings:
                # Append '/embeddings' without duplicating slashes
                new_path = f"{path_no_trailing}/embeddings"
                parsed = parsed._replace(path=new_path)
                return urlunparse(parsed)
            return api_url
        except Exception:
            # If parsing fails for any reason, fall back to the original URL.
            return api_url

    def _embed(self, text: str) -> List[float]:
        if not text:
            raise ValueError("Text to embed cannot be empty")
        headers = {"Content-Type": "application/json"}
        payload = {
            "input": text,
            "model": self.model_id,
            # TODO: Decide other parameters
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
