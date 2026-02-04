from typing import List

from .base import BaseIndexer

from cldk.analysis.java import JavaAnalysis
from sakura.nl2test.preprocessing.extractors import ClassSnippetExtractor, MethodSnippetExtractor
from sakura.nl2test.preprocessing.vector_stores import ProjectFAISSVectorStore
from sakura.nl2test.preprocessing.searchers import ProjectSearcher

"""DEPRECATED: Filter applies post search so often under-retrieves classes."""


class ProjectIndexer(BaseIndexer):
    def __init__(self, analysis: JavaAnalysis):
        super().__init__(analysis)
        self.class_extractor = ClassSnippetExtractor(analysis)
        self.method_extractor = MethodSnippetExtractor(analysis)

    def build_index(self, *, exclude_test_dirs: bool = False) -> ProjectSearcher:
        class_snippets = self.class_extractor.get_project_snippets(
            exclude_test_dirs=exclude_test_dirs
        )
        method_snippets = self.method_extractor.get_project_snippets(
            exclude_test_dirs=exclude_test_dirs
        )

        snippets: List = class_snippets + method_snippets
        vector_store = ProjectFAISSVectorStore(self.embedder)
        vector_store.add_snippets(snippets)
        return ProjectSearcher(vector_store)
