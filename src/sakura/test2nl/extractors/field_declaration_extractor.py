_A=None
from typing import List
from cldk.analysis.java import JavaAnalysis
from cldk.models.java.models import JField
from sakura.test2nl.model.models import FieldDeclaration
from sakura.utils.analysis import CommonAnalysis
class FieldDeclarationExtractor:
	def __init__(A,analysis:JavaAnalysis,application_classes:List[str]):A.analysis=analysis;A.application_classes=application_classes
	def _is_helper_class(A,type_name:str|_A)->bool|_A:
		'Check if type exists in repo but is not an application class.';B=type_name
		if not B:return
		D=CommonAnalysis.extract_non_parameterized_types(B)
		for C in D:
			if A.analysis.get_class(C)and C not in A.application_classes:return True
		return False
	def extract(C,field_declaration:JField)->FieldDeclaration:A=field_declaration;B=A.type if A.type else _A;return FieldDeclaration(variables=A.variables if A.variables else _A,type=B,modifiers=A.modifiers if A.modifiers else _A,annotations=A.annotations if A.annotations else _A,type_is_helper_class=C._is_helper_class(B))