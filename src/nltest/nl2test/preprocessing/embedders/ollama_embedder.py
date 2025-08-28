from typing import List, Optional

import numpy as np
import ollama

from .base_embedder import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    def __init__(self, model_id: str = "dengcao/Qwen3-Embedding-4B:Q5_K_M", dim: Optional[int] = None):
        self.model_id = model_id

        if dim is None:
            probe = self._raw_embed("probe")
            dim = int(probe.shape[0])
        super().__init__(dim)

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
            raise ValueError(f"Ollama returned a vector of length {arr.shape[0]}, expected {self.dim}")
        return arr.tolist()

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
