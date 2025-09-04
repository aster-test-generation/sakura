from abc import ABC, abstractmethod
from typing import Any, List

from nltest.nl2test.preprocessing.embedders import BaseEmbedder

class BaseVectorStore(ABC):
    def __init__(self, store, embedder: BaseEmbedder):
        self.store = store
        self.embedder = embedder

    @abstractmethod
    def add_snippets(self, snippets: List[Any]) -> None:
        """
        Embed and add each snippet to the vector store.
        Args:
            snippets: Raw items to embed and index
        """
        ...

    @abstractmethod
    def find_similar(self, query: str, k: int) -> List[Any]:
        """
        Return the k most similar items in the store to the query.
        Args:
            query: Text to embed and compare
            k: How many results to return
        """
        ...
