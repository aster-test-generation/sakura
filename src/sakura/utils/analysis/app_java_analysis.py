from __future__ import annotations
_B=False
_A=None
from pathlib import Path
from typing import Dict,List,Tuple
from cldk.analysis.java import JavaAnalysis
from cldk.models.java import JCallable
from cldk.models.java.models import JComment,JField,JMethodDetail,JType
class AppJavaAnalysis(JavaAnalysis):
	"\n    Wrapper over CLDK's JavaAnalysis that blocks out all non-application classes from \n    retrieval, effectively making existing test suites invisible.\n    "
	def __init__(A,*,application_classes:list[str],project_dir:str|Path|_A,source_code:str|_A,analysis_backend_path:str|_A,analysis_json_path:str|Path|_A,analysis_level:str,target_files:list[str]|_A,eager_analysis:bool)->_A:
		super().__init__(project_dir=project_dir,source_code=source_code,analysis_backend_path=analysis_backend_path,analysis_json_path=analysis_json_path,analysis_level=analysis_level,target_files=target_files,eager_analysis=eager_analysis);(A._allowed_classes):set[str]=set(application_classes)
		for B in list(A._allowed_classes):
			if'$'in B:A._allowed_classes.add(B.replace('$','.'))
	def _is_allowed_class(B,qualified_class_name:str)->bool:
		A=qualified_class_name
		if not A:return _B
		if A in B._allowed_classes:return True
		C=A.replace('$','.');return C in B._allowed_classes
	def get_classes(B)->Dict[str,JType]:A=super().get_classes();return{A:C for(A,C)in A.items()if B._is_allowed_class(A)}
	def get_classes_by_criteria(B,inclusions:list[str]|_A=_A,exclusions:list[str]|_A=_A)->Dict[str,JType]:A=super().get_classes_by_criteria(inclusions=inclusions,exclusions=exclusions);return{A:C for(A,C)in A.items()if B._is_allowed_class(A)}
	def get_methods(B)->Dict[str,Dict[str,JCallable]]:A=super().get_methods();return{A:C for(A,C)in A.items()if B._is_allowed_class(A)}
	def get_entry_point_classes(B)->Dict[str,JType]:A=super().get_entry_point_classes();return{A:C for(A,C)in A.items()if B._is_allowed_class(A)}
	def get_entry_point_methods(B)->Dict[str,Dict[str,JCallable]]:A=super().get_entry_point_methods();return{A:C for(A,C)in A.items()if B._is_allowed_class(A)}
	def get_class(B,qualified_class_name:str)->JType|_A:
		A=qualified_class_name
		if not B._is_allowed_class(A):return
		return super().get_class(A)
	def get_method(B,qualified_class_name:str,qualified_method_name:str)->JCallable|_A:
		A=qualified_class_name
		if not B._is_allowed_class(A):return
		return super().get_method(A,qualified_method_name)
	def get_method_parameters(B,qualified_class_name:str,qualified_method_name:str)->List[str]:
		A=qualified_class_name
		if not B._is_allowed_class(A):return[]
		return super().get_method_parameters(A,qualified_method_name)
	def get_java_file(B,qualified_class_name:str)->str|_A:
		A=qualified_class_name
		if not B._is_allowed_class(A):return
		return super().get_java_file(A)
	def get_methods_in_class(B,qualified_class_name:str)->Dict[str,JCallable]:
		A=qualified_class_name
		if not B._is_allowed_class(A):return{}
		return super().get_methods_in_class(A)
	def get_constructors(B,qualified_class_name:str)->Dict[str,JCallable]:
		A=qualified_class_name
		if not B._is_allowed_class(A):return{}
		return super().get_constructors(A)
	def get_fields(B,qualified_class_name:str)->List[JField]:
		A=qualified_class_name
		if not B._is_allowed_class(A):return[]
		return super().get_fields(A)
	def get_nested_classes(B,qualified_class_name:str)->List[JType]:
		A=qualified_class_name
		if not B._is_allowed_class(A):return[]
		return super().get_nested_classes(A)
	def get_sub_classes(B,qualified_class_name:str)->Dict[str,JType]:
		A=qualified_class_name
		if not B._is_allowed_class(A):return{}
		return super().get_sub_classes(A)
	def get_extended_classes(B,qualified_class_name:str)->List[str]:
		A=qualified_class_name
		if not B._is_allowed_class(A):return[]
		return super().get_extended_classes(A)
	def get_implemented_interfaces(B,qualified_class_name:str)->List[str]:
		A=qualified_class_name
		if not B._is_allowed_class(A):return[]
		return super().get_implemented_interfaces(A)
	def get_callers(B,target_class_name:str,target_method_declaration:str,using_symbol_table:bool=_B)->dict[str,object]:
		A=target_class_name
		if not B._is_allowed_class(A):return{}
		return super().get_callers(A,target_method_declaration,using_symbol_table)
	def get_callees(B,source_class_name:str,source_method_declaration:str,using_symbol_table:bool=_B)->dict[str,object]:
		A=source_class_name
		if not B._is_allowed_class(A):return{}
		return super().get_callees(A,source_method_declaration,using_symbol_table)
	def get_class_call_graph(B,qualified_class_name:str,method_signature:str|_A=_A,using_symbol_table:bool=_B)->List[Tuple[JMethodDetail,JMethodDetail]]:
		A=qualified_class_name
		if not B._is_allowed_class(A):return[]
		return super().get_class_call_graph(A,method_signature=method_signature,using_symbol_table=using_symbol_table)
	def get_comments_in_a_method(B,qualified_class_name:str,method_signature:str)->List[JComment]:
		A=qualified_class_name
		if not B._is_allowed_class(A):return[]
		return super().get_comments_in_a_method(A,method_signature)
	def get_comments_in_a_class(B,qualified_class_name:str)->List[JComment]:
		A=qualified_class_name
		if not B._is_allowed_class(A):return[]
		return super().get_comments_in_a_class(A)