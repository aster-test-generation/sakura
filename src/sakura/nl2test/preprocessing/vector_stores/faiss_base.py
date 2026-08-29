from __future__ import annotations

from pathlib import Path
from typing import Generic, List, Tuple, Protocol, TypeVar

import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from .base import BaseVectorStore

from sakura.nl2test.preprocessing.embedders import BaseEmbedder
from sakura.utils.pretty.color_logger import RichLog

T = TypeVar("T")


class SnippetConverter(Protocol[T]):
    """Generic callable that converts a snippet model into a LangChain Document."""

    def __call__(self, snippet: T) -> Document: ...


class BaseFAISSVectorStore(BaseVectorStore, Generic[T]):
    def __init__(
        self,
        embedder: BaseEmbedder,
        converter: SnippetConverter[T],
        *,
        index_dir: Path | None = None,
        use_stored_index: bool = False,
    ):
        self._converter = converter
        self._index_dir = index_dir
        self._use_stored_index = use_stored_index
        self._loaded_from_cache = False

        store = self._try_load_cached_store(embedder)
        if store is None:
            store = self._build_store(embedder)

        super().__init__(store, embedder)

    @property
    def loaded_from_cache(self) -> bool:
        return self._loaded_from_cache

    def _build_store(self, embedder: BaseEmbedder) -> FAISS:
        index = faiss.IndexFlatL2(embedder.dim)
        return FAISS(
            embedding_function=embedder,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
        )

    def _try_load_cached_store(self, embedder: BaseEmbedder) -> FAISS | None:
        if not self._use_stored_index:
            return None

        if self._index_dir is None:
            RichLog.warn(
                "use_stored_index was requested but index directory is not set."
            )
            return None

        if not self._index_dir.exists():
            return None
        if not self._index_dir.is_dir():
            RichLog.warn(
                f"Expected directory for FAISS index, got file: {self._index_dir}"
            )
            return None

        try:
            store = FAISS.load_local(
                str(self._index_dir),
                embedder,
                allow_dangerous_deserialization=True,
            )
            if store.index.d != embedder.dim:
                raise ValueError(
                    "Cached FAISS index dimension "
                    f"{store.index.d} does not match embedder dimension {embedder.dim}"
                )
            self._loaded_from_cache = True
            RichLog.debug(f"Loaded cached FAISS index from {self._index_dir}")
            return store
        except (
            FileNotFoundError,
            ValueError,
            RuntimeError,
            EOFError,
            AttributeError,
        ) as exc:
            RichLog.warn(
                f"Failed to load cached FAISS index at {self._index_dir}: {exc}"
            )
            self._loaded_from_cache = False
            return None

    def _save_index(self) -> None:
        if self._index_dir is None:
            return
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self.store.save_local(str(self._index_dir))
        RichLog.debug(f"Saved FAISS index to {self._index_dir}")

    def add_snippets(self, snippets: List[T]) -> None:
        if self._loaded_from_cache:
            # Cached index already populated; avoid duplicating entries.
            return

        if not snippets:
            self._save_index()
            return

        docs = [self._converter(s) for s in snippets]
        self.store.add_documents(docs)
        self._save_index()

    def find_similar(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        if k <= 0:
            raise ValueError("k must be positive")
        return self.store.similarity_search_with_score(query, k=k)
