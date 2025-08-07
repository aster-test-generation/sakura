from abc import ABC, abstractmethod
from typing import List
from langchain_core.embeddings import Embeddings

class BaseEmbedder(Embeddings, ABC):
    def __init__(self, dim: int):
        super().__init__()
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = dim

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        ...
