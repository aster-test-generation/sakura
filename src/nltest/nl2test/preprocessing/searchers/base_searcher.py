from __future__ import annotations
from typing import Generic, List, Protocol, TypeVar, Tuple, Optional

from langchain.schema import Document

VS = TypeVar("VS")


class _HasFindSimilar(Protocol):
    def find_similar(
            self,
            query: str,
            k: int = 5,
            **kwargs,
    ) -> List[Tuple[Document, float]]:
        ...


class BaseSearcher(Generic[VS]):
    def __init__(self, vector_store: VS):
        self._vs: _HasFindSimilar = vector_store

    def _doc_to_result(self, doc: Document) -> dict:
        raise NotImplementedError

    def find_similar(
            self,
            query: str,
            k: int = 5,
            **kwargs,
    ) -> List[dict]:
        return [
            self._doc_to_result(d)
            for d, _ in self._vs.find_similar(query, k, **kwargs)
        ]

    def find_similar_in_range(
            self,
            query: str,
            i: int,
            j: int,
            **kwargs
    ) -> List[dict]:
        hits = self._vs.find_similar(query, j, **kwargs)
        return [] if len(hits) < i else [
            self._doc_to_result(d) for d, _ in hits[i - 1: j]
        ]
