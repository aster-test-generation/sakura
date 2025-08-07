from typing import List

import numpy as np
import ollama

from .base_embedder import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    def __init__(self, model_id: str = "dengcao/Qwen3-Embedding-4B:Q5_K_M"):
        self.model_id = model_id

        # Probe for embedding dimension
        probe_embedding = self._embed("probe")
        dim = len(probe_embedding)
        super().__init__(dim)

    def _embed(self, text: str) -> List[float]:
        if not text:
            raise ValueError("Text to embed cannot be empty")
        try:
            raw = ollama.embeddings(model=self.model_id, prompt=text)["embedding"]
            arr = np.asarray(raw, dtype=np.float32).reshape(-1)
            if arr.shape[0] != self.dim:
                raise ValueError(f"Ollama returned a vector of length {arr.shape[0]}, expected {self.dim}")
            return arr.tolist()
        except Exception as e:
            raise RuntimeError(f"Error embedding with Ollama: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        batch = [self._embed(t) for t in texts]
        arr = np.asarray(batch, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(f"Batch embeddings have invalid shape {arr.shape}, expected (N, {self.dim})")
        return arr.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)
