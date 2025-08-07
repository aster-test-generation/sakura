from __future__ import annotations
from typing import Generic, List, Tuple, Protocol, TypeVar

import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from .base_vector_store import BaseVectorStore

from nltest.nl2test.preprocessing.embedders import BaseEmbedder

T = TypeVar("T")


class SnippetConverter(Protocol[T]):
    """Generic callable that converts a snippet model into a LangChain Document."""

    def __call__(self, snippet: T) -> Document:
        ...


class BaseFAISSVectorStore(BaseVectorStore, Generic[T]):
    def __init__(self, embedder: BaseEmbedder, converter: SnippetConverter[T]):
        self._converter = converter
        index = faiss.IndexFlatL2(embedder.dim)
        store = FAISS(
            embedding_function=embedder,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
        )
        super().__init__(store, embedder)

    def add_snippets(self, snippets: List[T]) -> None:
        if snippets:
            docs = [self._converter(s) for s in snippets]
            self.store.add_documents(docs)

    def find_similar(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        if k <= 0:
            raise ValueError("k must be positive")
        return self.store.similarity_search_with_score(query, k=k)
