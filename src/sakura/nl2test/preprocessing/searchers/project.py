from typing import List
from langchain.schema import Document
from.base import BaseSearcher
from sakura.nl2test.models import SnippetType
from sakura.nl2test.preprocessing.vector_stores import ProjectFAISSVectorStore
'DEPRECATED: Filter applies post search so often under-retrieves classes.'
class ProjectSearcher(BaseSearcher[ProjectFAISSVectorStore]):
	def _doc_to_result(H,doc:Document)->dict:
		G='method_signature';F='containing_class_name';E='type';D='snippet_type';B='declaring_class_name';A=doc
		try:C=SnippetType(A.metadata[D])
		except:raise RuntimeError(f"Unsupported snippet type: {A.metadata[D]}")
		if C==SnippetType.METHOD:return{E:'method',B:A.metadata[B],F:A.metadata[F],G:A.metadata[G]}
		if C==SnippetType.CLASS:return{E:'class',B:A.metadata[B]}
		raise Exception('Unknown snippet type: '+C.value)
	def find_methods(A,query:str,k:int=5)->List[dict]:return A.find_similar(query,k,snippet_type=SnippetType.METHOD)
	def find_classes(A,query:str,k:int=5)->List[dict]:return A.find_similar(query,k,snippet_type=SnippetType.CLASS)
	def find_methods_in_range(A,query:str,i:int,j:int)->List[dict]:return A.find_similar_in_range(query,i,j,snippet_type=SnippetType.METHOD)
	def find_classes_in_range(A,query:str,i:int,j:int)->List[dict]:return A.find_similar_in_range(query,i,j,snippet_type=SnippetType.CLASS)