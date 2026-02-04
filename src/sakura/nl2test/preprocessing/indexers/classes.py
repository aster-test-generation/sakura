from typing import List
from cldk.analysis.java import JavaAnalysis
from.base import BaseIndexer
from sakura.nl2test.models import ClassSnippet
from sakura.nl2test.preprocessing.extractors import ClassSnippetExtractor
from sakura.nl2test.preprocessing.searchers import ClassSearcher
from sakura.nl2test.preprocessing.vector_stores import ClassVectorStore
from sakura.utils.analysis import CommonAnalysis,Reachability
class ClassIndexer(BaseIndexer):
	def __init__(B,analysis:JavaAnalysis):A=analysis;super().__init__(A);B.extractor=ClassSnippetExtractor(A)
	def _get_classes_with_visible_methods(A):
		C=[]
		for B in A.analysis.get_classes():
			D=CommonAnalysis(A.analysis).get_testing_frameworks_for_class(B)
			if CommonAnalysis(A.analysis).is_test_class(B,D):continue
			if not Reachability(A.analysis).get_visible_class_methods(B):continue
			C.append(B)
		return C
	def build_index(B,*,exclude_test_dirs:bool=False)->ClassSearcher:
		'Extract snippets, embed them, add to vector store, and return searchers.';A=ClassVectorStore(B.embedder)
		if not A.loaded_from_cache:C:List[ClassSnippet]=B.extractor.get_project_snippets(exclude_test_dirs=exclude_test_dirs);A.add_snippets(C)
		return ClassSearcher(A)