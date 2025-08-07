from typing import List

from langchain.schema import Document

from .base_searcher import BaseSearcher

from nltest.nl2test.model.models import SnippetType
from nltest.nl2test.preprocessing.vector_stores import (
    ProjectFAISSVectorStore,
)

"""DEPRECATED: Filter applies post search so often under-retrieves classes."""


class ProjectSearcher(BaseSearcher[ProjectFAISSVectorStore]):
    def _doc_to_result(self, doc: Document) -> dict:
        try:
            snippet_type = SnippetType(doc.metadata["snippet_type"])
        except:
            raise RuntimeError(f"Unsupported snippet type: {doc.metadata['snippet_type']}")

        if snippet_type == SnippetType.METHOD:
            return {
                "type": "method",
                "qualified_class_name": doc.metadata["qualified_class_name"],
                "method_signature": doc.metadata["method_signature"],
            }

        if snippet_type == SnippetType.CLASS:
            return {
                "type": "class",
                "qualified_class_name": doc.metadata["qualified_class_name"],
            }

        raise Exception("Unknown snippet type: " + snippet_type.value)

    def find_methods(self, query: str, k: int = 5) -> List[dict]:
        return self.find_similar(query, k, snippet_type=SnippetType.METHOD)

    def find_classes(self, query: str, k: int = 5) -> List[dict]:
        return self.find_similar(query, k, snippet_type=SnippetType.CLASS)

    def find_methods_in_range(self, query: str, i: int, j: int) -> List[dict]:
        return self.find_similar_in_range(query, i, j, snippet_type=SnippetType.METHOD)

    def find_classes_in_range(self, query: str, i: int, j: int) -> List[dict]:
        return self.find_similar_in_range(query, i, j, snippet_type=SnippetType.CLASS)
