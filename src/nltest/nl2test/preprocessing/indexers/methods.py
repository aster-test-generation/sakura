from cldk.analysis.java import JavaAnalysis

from .base import BaseIndexer

from nltest.nl2test.preprocessing.extractors import MethodSnippetExtractor
from nltest.nl2test.preprocessing.vector_stores import MethodVectorStore
from nltest.nl2test.preprocessing.searchers import MethodSearcher


class MethodIndexer(BaseIndexer):
    def __init__(self, analysis: JavaAnalysis):
        super().__init__(analysis)
        self.extractor = MethodSnippetExtractor(analysis)

    def build_index(self, *, exclude_test_dirs: bool = False) -> MethodSearcher:
        """Extract snippets, embed them, add to vector store, and return searchers."""
        vector_store = MethodVectorStore(self.embedder)
        if not vector_store.loaded_from_cache:
            snippets = self.extractor.get_project_snippets(
                exclude_test_dirs=exclude_test_dirs
            )
            vector_store.add_snippets(snippets)
        return MethodSearcher(vector_store)
