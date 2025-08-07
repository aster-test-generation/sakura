from .base_searcher import BaseSearcher

from nltest.nl2test.preprocessing.vector_stores import ClassVectorStore


class ClassSearcher(BaseSearcher[ClassVectorStore]):
    def _doc_to_result(self, doc):
        return {
            "qualified_class_name": doc.metadata["qualified_class_name"]
        }
