from langchain.schema import Document

from .faiss_base import BaseFAISSVectorStore

from nltest.nl2test.preprocessing.embedders import BaseEmbedder
from nltest.nl2test.models import MethodSnippet


def _method_to_doc(snippet: MethodSnippet) -> Document:
    return Document(
        page_content=snippet.code,
        metadata={
            "implementing_class_name": snippet.implementing_class_name,
            "containing_class_name": snippet.containing_class_name,
            "method_signature": snippet.method_signature,
        },
    )


class MethodVectorStore(BaseFAISSVectorStore[MethodSnippet]):
    def __init__(self, embedder: BaseEmbedder):
        super().__init__(embedder, _method_to_doc)
