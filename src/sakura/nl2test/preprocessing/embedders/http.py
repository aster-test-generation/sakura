_B=True
_A=None
import os
from typing import List,Optional
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from tenacity import retry,stop_after_attempt,wait_exponential_jitter,retry_if_exception,RetryCallState
from.base import BaseEmbedder
from sakura.utils.pretty.color_logger import RichLog
EMBED_CHUNK_SIZE=100
DUMMY_API_KEY='EMPTY'
def _is_retriable_error(exc:BaseException)->bool:
	'Check if exception is transient and worth retrying.';C='status_code';A=exc;D=getattr(A,C,_A)or getattr(getattr(A,'response',_A),C,_A)
	if D in{429,500,502,503,504}:return _B
	if isinstance(A,(ConnectionError,TimeoutError)):return _B
	B=str(A).lower()
	if'rate limit'in B or'too many requests'in B or'overloaded'in B:return _B
	return False
def _log_retry_attempt(retry_state:RetryCallState)->_A:'Log retry attempts for observability.';A=retry_state;C=A.attempt_number;B=A.outcome.exception()if A.outcome else _A;D=A.next_action.sleep if A.next_action else 0;RichLog.warn(f"[HttpEmbedder] Retry attempt {C} after {D:.1f}s due to: {type(B).__name__ if B else'unknown'}")
class HttpEmbedder(BaseEmbedder):
	def __init__(B,model_id:str,api_url:str,api_key:Optional[str]=_A):
		C=api_key;D=C.strip()if C else _A;A:SecretStr|_A
		if D:A=SecretStr(D)
		elif os.getenv('OPENAI_API_KEY'):A=_A
		else:A=SecretStr(DUMMY_API_KEY);RichLog.warn('[HttpEmbedder] No embedding API key set; using placeholder value.')
		B._client=OpenAIEmbeddings(model=model_id,base_url=api_url.rstrip('/'),api_key=A,check_embedding_ctx_length=False);E=B._embed_query_with_retry('probe');super().__init__(len(E))
	@retry(stop=stop_after_attempt(3),wait=wait_exponential_jitter(initial=1,max=30),retry=retry_if_exception(_is_retriable_error),before_sleep=_log_retry_attempt,reraise=_B)
	def _embed_chunk_with_retry(self,texts:List[str])->List[List[float]]:'Embed a chunk of texts with retry logic.';return self._client.embed_documents(texts)
	@retry(stop=stop_after_attempt(3),wait=wait_exponential_jitter(initial=1,max=30),retry=retry_if_exception(_is_retriable_error),before_sleep=_log_retry_attempt,reraise=_B)
	def _embed_query_with_retry(self,text:str)->List[float]:'Embed a single query with retry logic.';return self._client.embed_query(text)
	def embed_documents(B,texts:List[str])->List[List[float]]:
		A=texts
		if not A:return[]
		if len(A)<=EMBED_CHUNK_SIZE:return B._embed_chunk_with_retry(A)
		C:List[List[float]]=[]
		for D in range(0,len(A),EMBED_CHUNK_SIZE):E=A[D:D+EMBED_CHUNK_SIZE];C.extend(B._embed_chunk_with_retry(E))
		return C
	def embed_query(A,text:str)->List[float]:return A._embed_query_with_retry(text)