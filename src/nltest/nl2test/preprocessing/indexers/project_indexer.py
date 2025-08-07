from typing import List

from .base_indexer import BaseIndexer

from cldk.analysis.java import JavaAnalysis
from nltest.nl2test.preprocessing.extractors import ClassSnippetExtractor, MethodSnippetExtractor
from nltest.nl2test.preprocessing.vector_stores import ProjectFAISSVectorStore
from nltest.nl2test.preprocessing.searchers import ProjectSearcher

"""DEPRECATED: Filter applies post search so often under-retrieves classes."""


class ProjectIndexer(BaseIndexer):
    def __init__(self, analysis: JavaAnalysis):
        super().__init__(analysis)
        self.class_extractor = ClassSnippetExtractor(analysis)
        self.method_extractor = MethodSnippetExtractor(analysis)

    def build_index(self) -> ProjectSearcher:
        class_snippets = self.class_extractor.get_project_snippets()
        method_snippets = self.method_extractor.get_project_snippets()

        snippets: List = class_snippets + method_snippets
        vector_store = ProjectFAISSVectorStore(self.embedder)
        vector_store.add_snippets(snippets)
        return ProjectSearcher(vector_store)
