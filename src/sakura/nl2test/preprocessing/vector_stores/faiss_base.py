from __future__ import annotations
_C=False
_B=True
_A=None
from pathlib import Path
from typing import Generic,List,Tuple,Protocol,TypeVar
import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from.base import BaseVectorStore
from sakura.nl2test.preprocessing.embedders import BaseEmbedder
from sakura.utils.pretty.color_logger import RichLog
T=TypeVar('T')
class SnippetConverter(Protocol[T]):
	'Generic callable that converts a snippet model into a LangChain Document.'
	def __call__(A,snippet:T)->Document:...
class BaseFAISSVectorStore(BaseVectorStore,Generic[T]):
	def __init__(A,embedder:BaseEmbedder,converter:SnippetConverter[T],*,index_dir:Path|_A=_A,use_stored_index:bool=_C):
		B=embedder;A._converter=converter;A._index_dir=index_dir;A._use_stored_index=use_stored_index;A._loaded_from_cache=_C;C=A._try_load_cached_store(B)
		if C is _A:C=A._build_store(B)
		super().__init__(C,B)
	@property
	def loaded_from_cache(self)->bool:return self._loaded_from_cache
	def _build_store(C,embedder:BaseEmbedder)->FAISS:A=embedder;B=faiss.IndexFlatL2(A.dim);return FAISS(embedding_function=A,index=B,docstore=InMemoryDocstore({}),index_to_docstore_id={})
	def _try_load_cached_store(A,embedder:BaseEmbedder)->FAISS|_A:
		if not A._use_stored_index:return
		if A._index_dir is _A:RichLog.warn('use_stored_index was requested but index directory is not set.');return
		if not A._index_dir.exists():return
		if not A._index_dir.is_dir():RichLog.warn(f"Expected directory for FAISS index, got file: {A._index_dir}");return
		try:B=FAISS.load_local(str(A._index_dir),embedder,allow_dangerous_deserialization=_B);A._loaded_from_cache=_B;RichLog.debug(f"Loaded cached FAISS index from {A._index_dir}");return B
		except(FileNotFoundError,ValueError,RuntimeError,EOFError,AttributeError,faiss.FaissError)as C:RichLog.warn(f"Failed to load cached FAISS index at {A._index_dir}: {C}");A._loaded_from_cache=_C;return
	def _save_index(A)->_A:
		if A._index_dir is _A:return
		A._index_dir.mkdir(parents=_B,exist_ok=_B);A.store.save_local(str(A._index_dir));RichLog.debug(f"Saved FAISS index to {A._index_dir}")
	def add_snippets(A,snippets:List[T])->_A:
		B=snippets
		if A._loaded_from_cache:return
		if not B:A._save_index();return
		C=[A._converter(B)for B in B];A.store.add_documents(C);A._save_index()
	def find_similar(A,query:str,k:int=5)->List[Tuple[Document,float]]:
		if k<=0:raise ValueError('k must be positive')
		return A.store.similarity_search_with_score(query,k=k)