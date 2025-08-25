from __future__ import annotations
from typing import Iterable, List, Tuple, Any, Dict

import faiss
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore

from .base_vector_store import BaseVectorStore

from nltest.nl2test.preprocessing.embedders import BaseEmbedder
from nltest.nl2test.model.models import SnippetType, Snippet, MethodSnippet, ClassSnippet

"""DEPRECATED: Filter applies post search so often under-retrieves classes."""


class ProjectFAISSVectorStore(BaseVectorStore):
    def __init__(self, embedder: BaseEmbedder):
        index = faiss.IndexFlatL2(embedder.dim)
        docstore = InMemoryDocstore({})
        index_to_docstore_id: Dict[int, str] = {}

        store = FAISS(
            embedding_function=embedder,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )
        super().__init__(store, embedder)

    def add_snippets(self, snippets: Iterable[Any]) -> None:
        docs = [self._to_doc(s) for s in snippets]
        if docs:
            self.store.add_documents(docs)

    def find_similar(
            self,
            query: str,
            k: int = 5,
            **kwargs,
    ) -> List[Tuple[Document, float]]:
        filter_ = {}

        snippet_type = kwargs.get("snippet_type")
        if snippet_type and isinstance(snippet_type, SnippetType):
            filter_["snippet_type"] = snippet_type.value

        # If empty dictionary, convert to None to avoid filter overhead
        if not filter_:
            filter_ = None

        return self.store.similarity_search_with_score(query, k=k, filter=filter_)

    def _to_doc(self, snippet: Snippet) -> Document:
        if isinstance(snippet, MethodSnippet):
            return Document(
                page_content=snippet.code,
                metadata={
                    "snippet_type": "method",
                    "implementing_class_name": snippet.implementing_class_name,
                    "containing_class_name": snippet.containing_class_name,
                    "method_signature": snippet.method_signature,
                },
            )

        if isinstance(snippet, ClassSnippet):
            return Document(
                page_content=snippet.simple_class_name,
                metadata={
                    "snippet_type": "class",
                    "implementing_class_name": snippet.implementing_class_name,
                },
            )

        raise Exception(f"Unsupported snippet type: {type(snippet)}")
