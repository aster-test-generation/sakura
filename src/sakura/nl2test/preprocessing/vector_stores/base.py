from abc import ABC,abstractmethod
from typing import Any,List
from sakura.nl2test.preprocessing.embedders import BaseEmbedder
class BaseVectorStore(ABC):
	def __init__(A,store,embedder:BaseEmbedder):A.store=store;A.embedder=embedder
	@abstractmethod
	def add_snippets(self,snippets:List[Any])->None:'\n        Embed and add each snippet to the vector store.\n        Args:\n            snippets: Raw items to embed and index\n        ';...
	@abstractmethod
	def find_similar(self,query:str,k:int)->List[Any]:'\n        Return the k most similar items in the store to the query.\n        Args:\n            query: Text to embed and compare\n            k: How many results to return\n        ';...