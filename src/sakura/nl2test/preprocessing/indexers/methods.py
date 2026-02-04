from cldk.analysis.java import JavaAnalysis
from.base import BaseIndexer
from sakura.nl2test.preprocessing.extractors import MethodSnippetExtractor
from sakura.nl2test.preprocessing.vector_stores import MethodVectorStore
from sakura.nl2test.preprocessing.searchers import MethodSearcher
class MethodIndexer(BaseIndexer):
	def __init__(B,analysis:JavaAnalysis):A=analysis;super().__init__(A);B.extractor=MethodSnippetExtractor(A)
	def build_index(B,*,exclude_test_dirs:bool=False)->MethodSearcher:
		'Extract snippets, embed them, add to vector store, and return searchers.';A=MethodVectorStore(B.embedder)
		if not A.loaded_from_cache:C=B.extractor.get_project_snippets(exclude_test_dirs=exclude_test_dirs);A.add_snippets(C)
		return MethodSearcher(A)