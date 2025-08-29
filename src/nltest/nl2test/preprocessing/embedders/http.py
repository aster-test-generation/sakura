from typing import List

import requests

from .base import BaseEmbedder


class HttpEmbedder(BaseEmbedder):
    def __init__(self, model_id: str, api_url: str):
        self.model_id = model_id
        self.api_url = api_url

        # Probe for embedding dimension
        probe_embedding = self._embed("probe")
        dim = len(probe_embedding)
        super().__init__(dim)

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
