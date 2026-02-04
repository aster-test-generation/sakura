from typing import List

from cldk.analysis.java import JavaAnalysis

from .base import BaseIndexer

from sakura.nl2test.models import ClassSnippet
from sakura.nl2test.preprocessing.extractors import ClassSnippetExtractor
from sakura.nl2test.preprocessing.searchers import ClassSearcher
from sakura.nl2test.preprocessing.vector_stores import ClassVectorStore
from sakura.utils.analysis import CommonAnalysis, Reachability


class ClassIndexer(BaseIndexer):
    def __init__(self, analysis: JavaAnalysis):
        super().__init__(analysis)
        self.extractor = ClassSnippetExtractor(analysis)

    def _get_classes_with_visible_methods(self):
        classes = []
        for qualified_class_name in self.analysis.get_classes():
            testing_frameworks = CommonAnalysis(
                self.analysis
            ).get_testing_frameworks_for_class(qualified_class_name)
            if CommonAnalysis(self.analysis).is_test_class(
                qualified_class_name, testing_frameworks
            ):
                continue
            if not Reachability(self.analysis).get_visible_class_methods(
                qualified_class_name
            ):
                continue
            classes.append(qualified_class_name)
        return classes

    def build_index(self, *, exclude_test_dirs: bool = False) -> ClassSearcher:
        """Extract snippets, embed them, add to vector store, and return searchers."""
        vector_store = ClassVectorStore(self.embedder)
        if not vector_store.loaded_from_cache:
            snippets: List[ClassSnippet] = self.extractor.get_project_snippets(
                exclude_test_dirs=exclude_test_dirs
            )
            vector_store.add_snippets(snippets)
        return ClassSearcher(vector_store)
