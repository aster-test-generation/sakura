from typing import List, Tuple, Optional, Dict

from cldk.analysis.java import JavaAnalysis

from nltest.nl2test.model.models import MethodSnippet, ClassSnippet
from nltest.utils.analysis.common_analysis import CommonAnalysis
from nltest.utils.analysis.reachability import Reachability


class ClassSnippetExtractor:
    def __init__(self, analysis: JavaAnalysis):
        self.analysis = analysis

    def get_class_snippets(self):
        """Get the non-test classes with visible methods."""
        class_snippets: List[ClassSnippet] = []
        for qualified_class_name in self.analysis.get_classes():
            testing_frameworks = CommonAnalysis(self.analysis).get_testing_frameworks_for_class(qualified_class_name)
            if CommonAnalysis(self.analysis).is_test_class(qualified_class_name, testing_frameworks):
                continue
            if not Reachability(self.analysis).get_reachable_class_methods(qualified_class_name):
                continue
            class_snippets.append(
                ClassSnippet(
                    simple_class_name=qualified_class_name.split(".")[-1],
                    qualified_class_name=qualified_class_name,
                )
            )
        return class_snippets

    def get_project_snippets(self) -> List[ClassSnippet]:
        class_snippets: List[ClassSnippet] = self.get_class_snippets()
        return class_snippets
