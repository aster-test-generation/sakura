from typing import List, Tuple, Optional, Dict

from cldk.analysis.java import JavaAnalysis

from nltest.nl2test.model.models import MethodSnippet
from nltest.utils.analysis import CommonAnalysis, Reachability


class MethodSnippetExtractor:
    def __init__(self, analysis: JavaAnalysis):
        self.analysis = analysis

    def _reconstruct_simple_class_declaration(self, qualified_class_name: str) -> str:
        simple_class_name = qualified_class_name.split(".")[-1]
        class_details = self.analysis.get_class(qualified_class_name)
        if not class_details:
            return f"public class {simple_class_name}"

        parts: List[str] = []

        # Can do annotations here later if necessary

        if class_details.modifiers:
            parts.append(" ".join(class_details.modifiers))

        if class_details.is_enum_declaration:
            type_keyword = "enum"
        elif class_details.is_annotation_declaration:
            type_keyword = "@interface"
        elif class_details.is_record_declaration:
            type_keyword = "record"
        elif class_details.is_interface:
            type_keyword = "interface"
        else:
            type_keyword = "class"
        parts.append(type_keyword)

        parts.append(simple_class_name)

        # Can include extends and implements here if necessary later

        declaration = " ".join(parts)
        return declaration

    def _format_code(self, qualified_class_name: str, method_signature: str, containing_class: str) -> str:
        method_details = self.analysis.get_method(qualified_class_name, method_signature)
        code = CommonAnalysis.get_complete_method_code(method_details.declaration, method_details.code)
        simple_class_decl = self._reconstruct_simple_class_declaration(
            containing_class)  # Containing class for declaration
        return simple_class_decl + "{\n" + code + "\n}"

    def get_method_snippet(self, qualified_class_name: str, method_signature: str,
                           containing_class: Optional[str] = None) -> MethodSnippet | None:
        """
        Creates a method snippet from the method signature and qualified class and returns it. The qualified class name
        refers to the actual class containing the method, but the containing class refers to the class under analysis
        (if the method is inherited)
        Args:
            qualified_class_name:
            method_signature:
            containing_class:

        Returns:

        """
        if not self.analysis.get_method(qualified_class_name, method_signature):
            return None
        if not containing_class:
            containing_class = qualified_class_name
        code = self._format_code(qualified_class_name, method_signature, containing_class)
        return MethodSnippet(
            implementing_class_name=qualified_class_name, # NOTE: This is the class that contains the method, not the class that the method is in
            method_signature=method_signature,
            code=code, # NOTE: This contains the class of the containing class for retrieval
            containing_class_name=containing_class,
        )

    def get_class_snippets(self, qualified_class_name: str) -> List[MethodSnippet]:
        if not self.analysis.get_class(qualified_class_name):
            return []

        method_snippets: List[MethodSnippet] = []
        testing_frameworks = CommonAnalysis(self.analysis).get_testing_frameworks_for_class(qualified_class_name)

        # Do not process test classes
        if CommonAnalysis(self.analysis).is_test_class(qualified_class_name, testing_frameworks):
            return []

        # Get all reachable methods from the class
        reachable_methods: Dict[str, List[str]] = Reachability(self.analysis).get_visible_class_methods(
            qualified_class_name, visibility_mode="same_package_or_subclass")

        for cls, method_sigs in reachable_methods.items():
            for method_sig in method_sigs:
                method_snippet = self.get_method_snippet(cls, method_sig, containing_class=qualified_class_name)
                if method_snippet:
                    method_snippets.append(method_snippet)

        return method_snippets

    def get_project_snippets(self) -> List[MethodSnippet]:
        """Collect all non-test method snippets from the project."""
        method_snippets: List[MethodSnippet] = []
        for qualified_class in self.analysis.get_classes():
            method_snippets.extend(self.get_class_snippets(qualified_class))
        return method_snippets
