from .base_searcher import BaseSearcher

from nltest.nl2test.preprocessing.vector_stores import MethodVectorStore


class MethodSearcher(BaseSearcher[MethodVectorStore]):
    def _doc_to_result(self, doc):
        return {
            "implementing_class_name": doc.metadata["implementing_class_name"],
            "containing_class_name": doc.metadata["containing_class_name"],
            "method_signature": doc.metadata["method_signature"],
        }
