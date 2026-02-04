from typing import List
from.base import BaseIndexer
from cldk.analysis.java import JavaAnalysis
from sakura.nl2test.preprocessing.extractors import ClassSnippetExtractor,MethodSnippetExtractor
from sakura.nl2test.preprocessing.vector_stores import ProjectFAISSVectorStore
from sakura.nl2test.preprocessing.searchers import ProjectSearcher
'DEPRECATED: Filter applies post search so often under-retrieves classes.'
class ProjectIndexer(BaseIndexer):
	def __init__(B,analysis:JavaAnalysis):A=analysis;super().__init__(A);B.class_extractor=ClassSnippetExtractor(A);B.method_extractor=MethodSnippetExtractor(A)
	def build_index(A,*,exclude_test_dirs:bool=False)->ProjectSearcher:B=exclude_test_dirs;D=A.class_extractor.get_project_snippets(exclude_test_dirs=B);E=A.method_extractor.get_project_snippets(exclude_test_dirs=B);F:List=D+E;C=ProjectFAISSVectorStore(A.embedder);C.add_snippets(F);return ProjectSearcher(C)