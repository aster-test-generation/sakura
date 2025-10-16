from pathlib import Path
from langchain.schema import Document

from .faiss_base import BaseFAISSVectorStore

from nltest.nl2test.preprocessing.embedders import BaseEmbedder
from nltest.nl2test.models import ClassSnippet
from nltest.utils.config import Config
from nltest.utils.exceptions import ConfigurationException


def _class_to_doc(snippet: ClassSnippet) -> Document:
    return Document(
        page_content=snippet.simple_class_name,
        metadata={
            "implementing_class_name": snippet.implementing_class_name
        },
    )


class ClassVectorStore(BaseFAISSVectorStore[ClassSnippet]):
    def __init__(self, embedder: BaseEmbedder):
        config = Config()
        try:
            use_stored_index = bool(config.get("project", "use_stored_index"))
        except ConfigurationException:
            use_stored_index = False

        index_dir: Path | None = None
        try:
            project_output_dir = Path(config.get("project", "project_output_dir"))
            index_dir = project_output_dir / "indexes" / "class"
        except ConfigurationException:
            index_dir = None
            use_stored_index = False

        super().__init__(
            embedder,
            _class_to_doc,
            index_dir=index_dir,
            use_stored_index=use_stored_index,
        )
