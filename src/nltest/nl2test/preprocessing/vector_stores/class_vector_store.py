from langchain.schema import Document

from .base_faiss_vector_store import BaseFAISSVectorStore

from nltest.nl2test.preprocessing.embedders import BaseEmbedder
from nltest.nl2test.model.models import ClassSnippet


def _class_to_doc(snippet: ClassSnippet) -> Document:
    return Document(
        page_content=snippet.simple_class_name,
        metadata={
            "implementing_class_name": snippet.implementing_class_name
        },
    )


class ClassVectorStore(BaseFAISSVectorStore[ClassSnippet]):
    def __init__(self, embedder: BaseEmbedder):
        super().__init__(embedder, _class_to_doc)
