from .base_searcher import BaseSearcher

from nltest.nl2test.preprocessing.vector_stores import MethodVectorStore


class MethodSearcher(BaseSearcher[MethodVectorStore]):
    def _doc_to_result(self, doc):
        return {
            "qualified_class_name": doc.metadata["qualified_class_name"],
            "method_signature": doc.metadata["method_signature"],
        }
