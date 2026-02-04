from __future__ import annotations
from typing import Generic,List,Protocol,TypeVar,Tuple,Optional
from langchain.schema import Document
VS=TypeVar('VS')
class _HasFindSimilar(Protocol):
	def find_similar(A,query:str,k:int=5,**B)->List[Tuple[Document,float]]:...
class BaseSearcher(Generic[VS]):
	def __init__(A,vector_store:VS):(A._vs):_HasFindSimilar=vector_store
	def _doc_to_result(A,doc:Document)->dict:raise NotImplementedError
	def find_similar(A,query:str,k:int=5,**B)->List[dict]:return[A._doc_to_result(B)for(B,C)in A._vs.find_similar(query,k,**B)]
	def find_similar_in_range(A,query:str,i:int,j:int,**C)->List[dict]:B=A._vs.find_similar(query,j,**C);return[]if len(B)<i else[A._doc_to_result(B)for(B,C)in B[i-1:j]]