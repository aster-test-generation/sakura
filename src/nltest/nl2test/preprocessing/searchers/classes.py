from .base import BaseSearcher

from nltest.nl2test.preprocessing.vector_stores import ClassVectorStore


class ClassSearcher(BaseSearcher[ClassVectorStore]):
    def _doc_to_result(self, doc):
        return {
            "implementing_class_name": doc.metadata["implementing_class_name"]
        }
