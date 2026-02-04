from typing import List

from cldk.analysis.java import JavaAnalysis
from cldk.models.java.models import JField

from sakura.test2nl.model.models import FieldDeclaration
from sakura.utils.analysis import CommonAnalysis


class FieldDeclarationExtractor:
    def __init__(self, analysis: JavaAnalysis, application_classes: List[str]):
        self.analysis = analysis
        self.application_classes = application_classes

    def _is_helper_class(self, type_name: str | None) -> bool | None:
        """Check if type exists in repo but is not an application class."""
        if not type_name:
            return None
        non_param_types = CommonAnalysis.extract_non_parameterized_types(type_name)
        for type_ in non_param_types:
            if self.analysis.get_class(type_) and type_ not in self.application_classes:
                return True
        return False

    def extract(self, field_declaration: JField) -> FieldDeclaration:
        field_type = field_declaration.type if field_declaration.type else None
        return FieldDeclaration(
            variables=field_declaration.variables
            if field_declaration.variables
            else None,
            type=field_type,
            modifiers=field_declaration.modifiers
            if field_declaration.modifiers
            else None,
            annotations=field_declaration.annotations
            if field_declaration.annotations
            else None,
            type_is_helper_class=self._is_helper_class(field_type),
        )
