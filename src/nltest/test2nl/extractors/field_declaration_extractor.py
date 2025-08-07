from cldk.analysis.java import JavaAnalysis
from cldk.models.java.models import JField

from nltest.test2nl.model.models import FieldDeclaration


class FieldDeclarationExtractor:
    def __init__(self, analysis: JavaAnalysis):
        self.analysis = analysis

    @staticmethod
    def extract(field_declaration: JField) -> FieldDeclaration:
        return FieldDeclaration(
            variables=field_declaration.variables if field_declaration.variables else None,
            type=field_declaration.type if field_declaration.type else None,
            modifiers=field_declaration.modifiers if field_declaration.modifiers else None,
            annotations=field_declaration.annotations if field_declaration.annotations else None,
        )
