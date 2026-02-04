from abc import ABC,abstractmethod
from typing import List
from langchain_core.embeddings import Embeddings
class BaseEmbedder(Embeddings,ABC):
	def __init__(B,dim:int):
		A=dim;super().__init__()
		if not isinstance(A,int)or A<=0:raise ValueError('dim must be a positive integer')
		B.dim=A
	@abstractmethod
	def embed_documents(self,texts:List[str])->List[List[float]]:...
	@abstractmethod
	def embed_query(self,text:str)->List[float]:...