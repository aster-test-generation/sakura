from typing import List

from cldk.analysis.java import JavaAnalysis

from nltest.test2nl.model.models import ClassContext, FieldDeclaration, MethodContext

from .field_declaration_extractor import FieldDeclarationExtractor
from .method_extractor import MethodExtractor


class ClassExtractor:
    def __init__(self, analysis: JavaAnalysis, application_classes: List[str]):
        self.analysis = analysis
        self.application_classes = application_classes
        self.field_extractor = FieldDeclarationExtractor(analysis, application_classes)
        self.method_extractor = MethodExtractor(analysis, application_classes)

    def extract(
        self,
        qualified_class_name: str,
        complete_methods: bool,
        called_method_names: set[str] | None = None,
    ) -> ClassContext:
        class_file = self.analysis.get_java_file(qualified_class_name)
        if not class_file:
            raise RuntimeError(f"Could not find class file {qualified_class_name}")
        compilation_unit = self.analysis.get_java_compilation_unit(class_file)

        class_details = self.analysis.get_class(qualified_class_name)

        # Simple class name
        simple_class_name = qualified_class_name.rsplit(".", 1)[-1]

        # Get annotations for class
        annotations = class_details.annotations if class_details.annotations else None

        # Get extensions for class
        extends = class_details.extends_list if class_details.extends_list else None

        # Get modifiers for class
        modifiers = class_details.modifiers if class_details.modifiers else None

        # Get field declarations for the class that are PUBLIC
        field_declarations: List[FieldDeclaration] = []
        for field_declaration in class_details.field_declarations:
            if any(modifier == "public" for modifier in field_declaration.modifiers):
                field_declarations.append(
                    self.field_extractor.extract(field_declaration)
                )

        # Get methods for class (filtered if called_method_names provided)
        methods: List[MethodContext] = []
        method_sigs = [
            method_sig
            for method_sig in self.analysis.get_methods_in_class(qualified_class_name)
        ]
        for method_sig in self.analysis.get_methods_in_class(qualified_class_name):
            method_details = self.analysis.get_method(qualified_class_name, method_sig)
            if not method_details:
                continue
            method_name = method_sig.split("(")[0]
            is_constructor = method_details.is_constructor or (
                method_name == simple_class_name or method_name == "<init>"
            )
            if called_method_names is not None and not is_constructor:
                if method_name not in called_method_names:
                    continue
            methods.append(
                self.method_extractor.extract(
                    qualified_class_name, method_sig, complete_methods
                )
            )

        # Get comments for the compilation unit -> May contain semantic information about the class
        compilation_comments = [
            comment.content
            for comment in compilation_unit.comments
            if comment.content and comment.is_javadoc
        ]

        return ClassContext(
            simple_class_name=simple_class_name,
            qualified_class_name=qualified_class_name,
            annotations=annotations,
            extends=extends,
            modifiers=modifiers,
            field_declarations=field_declarations if field_declarations else None,
            relevant_class_methods=methods if methods else None,
            javadoc=compilation_comments if compilation_comments else None,
        )
