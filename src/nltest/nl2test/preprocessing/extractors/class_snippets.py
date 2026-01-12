from typing import List, Tuple, Optional, Dict

from cldk.analysis.java import JavaAnalysis

from nltest.nl2test.models import MethodSnippet, ClassSnippet
from nltest.utils.analysis.java_analyzer import CommonAnalysis, Reachability
from nltest.utils.constants import is_test_source_path


class ClassSnippetExtractor:
    def __init__(self, analysis: JavaAnalysis):
        self.analysis = analysis

    def get_class_snippets(self, exclude_test_dirs: bool = False) -> List[ClassSnippet]:
        """Get the non-test classes with visible methods."""
        class_snippets: List[ClassSnippet] = []
        for qualified_class_name in self.analysis.get_classes():
            if exclude_test_dirs and is_test_source_path(
                self.analysis.get_java_file(qualified_class_name)
            ):
                continue

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
            class_snippets.append(
                ClassSnippet(
                    simple_class_name=qualified_class_name.split(".")[-1],
                    declaring_class_name=qualified_class_name,
                )
            )
        return class_snippets

    def get_project_snippets(
        self, exclude_test_dirs: bool = False
    ) -> List[ClassSnippet]:
        class_snippets: List[ClassSnippet] = self.get_class_snippets(exclude_test_dirs)
        return class_snippets
