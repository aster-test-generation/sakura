from langchain.schema import Document

from .base_faiss_vector_store import BaseFAISSVectorStore

from nltest.nl2test.preprocessing.embedders import BaseEmbedder
from nltest.nl2test.model.models import MethodSnippet


def _method_to_doc(snippet: MethodSnippet) -> Document:
    return Document(
        page_content=snippet.code,
        metadata={
            "qualified_class_name": snippet.qualified_class_name,
            "method_signature": snippet.method_signature,
        },
    )


class MethodVectorStore(BaseFAISSVectorStore[MethodSnippet]):
    def __init__(self, embedder: BaseEmbedder):
        super().__init__(embedder, _method_to_doc)
