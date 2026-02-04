from .base import BaseSearcher

from sakura.nl2test.preprocessing.vector_stores import MethodVectorStore


class MethodSearcher(BaseSearcher[MethodVectorStore]):
    def _doc_to_result(self, doc):
        return {
            "declaring_class_name": doc.metadata["declaring_class_name"],
            "containing_class_name": doc.metadata["containing_class_name"],
            "method_signature": doc.metadata["method_signature"],
        }
