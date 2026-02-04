from typing import List,Tuple,Optional,Dict
from cldk.analysis.java import JavaAnalysis
from sakura.nl2test.models import MethodSnippet,ClassSnippet
from sakura.utils.analysis.java_analyzer import CommonAnalysis,Reachability
from sakura.utils.constants import is_test_source_path
class ClassSnippetExtractor:
	def __init__(A,analysis:JavaAnalysis):A.analysis=analysis
	def get_class_snippets(B,exclude_test_dirs:bool=False)->List[ClassSnippet]:
		'Get the non-test classes with visible methods.';C:List[ClassSnippet]=[]
		for A in B.analysis.get_classes():
			if exclude_test_dirs and is_test_source_path(B.analysis.get_java_file(A)):continue
			D=CommonAnalysis(B.analysis).get_testing_frameworks_for_class(A)
			if CommonAnalysis(B.analysis).is_test_class(A,D):continue
			if not Reachability(B.analysis).get_visible_class_methods(A):continue
			C.append(ClassSnippet(simple_class_name=A.split('.')[-1],declaring_class_name=A))
		return C
	def get_project_snippets(A,exclude_test_dirs:bool=False)->List[ClassSnippet]:B:List[ClassSnippet]=A.get_class_snippets(exclude_test_dirs);return B