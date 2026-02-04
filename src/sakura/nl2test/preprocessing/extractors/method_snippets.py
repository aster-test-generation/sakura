from typing import Dict,List,Optional
from cldk.analysis.java import JavaAnalysis
from sakura.nl2test.models import MethodSnippet
from sakura.utils.analysis import CommonAnalysis,Reachability
from sakura.utils.constants import is_test_source_path
class MethodSnippetExtractor:
	def __init__(A,analysis:JavaAnalysis):A.analysis=analysis
	def _reconstruct_simple_class_declaration(F,qualified_class_name:str)->str:
		D=qualified_class_name;E=D.split('.')[-1];A=F.analysis.get_class(D)
		if not A:return f"public class {E}"
		C:List[str]=[]
		if A.modifiers:C.append(' '.join(A.modifiers))
		if A.is_enum_declaration:B='enum'
		elif A.is_annotation_declaration:B='@interface'
		elif A.is_record_declaration:B='record'
		elif A.is_interface:B='interface'
		else:B='class'
		C.append(B);C.append(E);G=' '.join(C);return G
	def _format_code(A,qualified_class_name:str,method_signature:str,containing_class:str)->str:B=A.analysis.get_method(qualified_class_name,method_signature);C=CommonAnalysis.get_complete_method_code(B.declaration,B.code);D=A._reconstruct_simple_class_declaration(containing_class);return D+'{\n'+C+'\n}'
	def get_method_snippet(D,qualified_class_name:str,method_signature:str,containing_class:Optional[str]=None)->MethodSnippet|None:
		'\n        Creates a method snippet from the method signature and qualified class and returns it. The qualified class name\n        refers to the actual class containing the method, but the containing class refers to the class under analysis\n        (if the method is inherited)\n        Args:\n            qualified_class_name:\n            method_signature:\n            containing_class:\n\n        Returns:\n\n        ';C=method_signature;B=containing_class;A=qualified_class_name
		if not D.analysis.get_method(A,C):return
		if not B:B=A
		E=D._format_code(A,C,B);return MethodSnippet(declaring_class_name=A,method_signature=C,code=E,containing_class_name=B)
	def get_class_snippets(A,qualified_class_name:str,exclude_test_dirs:bool=False)->List[MethodSnippet]:
		B=qualified_class_name
		if not A.analysis.get_class(B):return[]
		if exclude_test_dirs and is_test_source_path(A.analysis.get_java_file(B)):return[]
		C:List[MethodSnippet]=[];E=CommonAnalysis(A.analysis).get_testing_frameworks_for_class(B)
		if CommonAnalysis(A.analysis).is_test_class(B,E):return[]
		F:Dict[str,List[str]]=Reachability(A.analysis).get_visible_class_methods(B,visibility_mode='same_package_or_subclass')
		for(G,H)in F.items():
			for I in H:
				D=A.get_method_snippet(G,I,containing_class=B)
				if D:C.append(D)
		return C
	def get_project_snippets(A,exclude_test_dirs:bool=False)->List[MethodSnippet]:
		'Collect all non-test method snippets from the project.';B:List[MethodSnippet]=[]
		for C in A.analysis.get_classes():B.extend(A.get_class_snippets(C,exclude_test_dirs))
		return B