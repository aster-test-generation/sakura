from __future__ import annotations
_A='snippet_type'
from typing import Iterable,List,Tuple,Any,Dict
import faiss
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from.base import BaseVectorStore
from sakura.nl2test.preprocessing.embedders import BaseEmbedder
from sakura.nl2test.models import SnippetType,Snippet,MethodSnippet,ClassSnippet
'DEPRECATED: Filter applies post search so often under-retrieves classes.'
class ProjectFAISSVectorStore(BaseVectorStore):
	def __init__(F,embedder:BaseEmbedder):A=embedder;B=faiss.IndexFlatL2(A.dim);C=InMemoryDocstore({});D:Dict[int,str]={};E=FAISS(embedding_function=A,index=B,docstore=C,index_to_docstore_id=D);super().__init__(E,A)
	def add_snippets(A,snippets:Iterable[Any])->None:
		B=[A._to_doc(B)for B in snippets]
		if B:A.store.add_documents(B)
	def find_similar(C,query:str,k:int=5,**D)->List[Tuple[Document,float]]:
		A={};B=D.get(_A)
		if B and isinstance(B,SnippetType):A[_A]=B.value
		if not A:A=None
		return C.store.similarity_search_with_score(query,k=k,filter=A)
	def _to_doc(C,snippet:Snippet)->Document:
		B='declaring_class_name';A=snippet
		if isinstance(A,MethodSnippet):return Document(page_content=A.code,metadata={_A:'method',B:A.declaring_class_name,'containing_class_name':A.containing_class_name,'method_signature':A.method_signature})
		if isinstance(A,ClassSnippet):return Document(page_content=A.simple_class_name,metadata={_A:'class',B:A.declaring_class_name})
		raise Exception(f"Unsupported snippet type: {type(A)}")