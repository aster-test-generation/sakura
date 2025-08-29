from cldk.analysis.java import JavaAnalysis

from .base import BaseIndexer

from nltest.nl2test.preprocessing.extractors import MethodSnippetExtractor
from nltest.nl2test.preprocessing.vector_stores import MethodVectorStore
from nltest.nl2test.preprocessing.searchers import MethodSearcher


class MethodIndexer(BaseIndexer):
    def __init__(self, analysis: JavaAnalysis):
        super().__init__(analysis)
        self.extractor = MethodSnippetExtractor(analysis)

    def build_index(self) -> MethodSearcher:
        """Extract snippets, embed them, add to vector store, and return searchers."""
        snippets = self.extractor.get_project_snippets()
        vector_store = MethodVectorStore(self.embedder)
        vector_store.add_snippets(snippets)
        return MethodSearcher(vector_store)
