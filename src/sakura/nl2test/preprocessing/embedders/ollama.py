_B=None
_A=True
from concurrent.futures import ThreadPoolExecutor
from typing import List,Optional
import numpy as np,ollama
from tenacity import retry,stop_after_attempt,wait_exponential_jitter,retry_if_exception,RetryCallState
from.base import BaseEmbedder
from sakura.utils.pretty.color_logger import RichLog
OLLAMA_MAX_WORKERS=4
def _is_retriable_error(exc:BaseException)->bool:
	'Check if exception is transient and worth retrying.'
	if isinstance(exc,(ConnectionError,TimeoutError)):return _A
	A=str(exc).lower()
	if'rate limit'in A or'too many requests'in A or'overloaded'in A:return _A
	if'connection'in A or'timeout'in A:return _A
	return False
def _log_retry_attempt(retry_state:RetryCallState)->_B:'Log retry attempts for observability.';A=retry_state;C=A.attempt_number;B=A.outcome.exception()if A.outcome else _B;D=A.next_action.sleep if A.next_action else 0;RichLog.warn(f"[OllamaEmbedder] Retry attempt {C} after {D:.1f}s due to: {type(B).__name__ if B else'unknown'}")
class OllamaEmbedder(BaseEmbedder):
	def __init__(B,model_id:str='dengcao/Qwen3-Embedding-4B:Q5_K_M',dim:Optional[int]=_B):
		A=dim;B.model_id=model_id
		if A is _B:C=B._raw_embed('probe');A=int(C.shape[0])
		super().__init__(A)
	@retry(stop=stop_after_attempt(3),wait=wait_exponential_jitter(initial=1,max=30),retry=retry_if_exception(_is_retriable_error),before_sleep=_log_retry_attempt,reraise=_A)
	def _raw_embed(self,text:str)->np.ndarray:
		if not text:raise ValueError('Text to embed cannot be empty')
		try:A=ollama.embeddings(model=self.model_id,prompt=text)['embedding']
		except Exception as B:raise RuntimeError(f"Error embedding with Ollama: {B}")
		return np.asarray(A,dtype=np.float32).reshape(-1)
	def _embed(A,text:str)->List[float]:
		B=A._raw_embed(text)
		if B.shape[0]!=A.dim:raise ValueError(f"Ollama returned a vector of length {B.shape[0]}, expected {A.dim}")
		return B.tolist()
	def embed_documents(B,texts:List[str])->List[List[float]]:
		C=texts
		if not C:return[]
		with ThreadPoolExecutor(max_workers=OLLAMA_MAX_WORKERS)as D:E=list(D.map(B._embed,C))
		A=np.asarray(E,dtype=np.float32)
		if A.ndim!=2 or A.shape[1]!=B.dim:raise ValueError(f"Batch embeddings have invalid shape {A.shape}, expected (N, {B.dim})")
		return A.tolist()
	def embed_query(A,text:str)->List[float]:return A._embed(text)