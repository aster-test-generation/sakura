from abc import ABC,abstractmethod
from cldk.analysis.java import JavaAnalysis
from sakura.nl2test.preprocessing.embedders import HttpEmbedder,OllamaEmbedder
from sakura.utils.config import Config
from sakura.utils.llm.model import Provider
class BaseIndexer(ABC):
	def __init__(A,analysis:JavaAnalysis):A.analysis=analysis;A.config=Config();A.embedder=A._initialize_embedder()
	def _initialize_embedder(A):
		C='emb';G=A.config.get(C,'provider');D=A.config.get(C,'model')
		try:B=Provider(G)
		except ValueError:B=None
		if B==Provider.OLLAMA:return OllamaEmbedder(model_id=D)
		else:
			E=A.config.get(C,'api_url')
			if not E:H=B.name if B else'unknown';raise ValueError(f"API URL is missing in config for HTTP provider {H}")
			try:F=A.config.get(C,'api_key')
			except Exception:F=None
			return HttpEmbedder(model_id=D,api_url=E,api_key=F)
	@abstractmethod
	def build_index(self,*,exclude_test_dirs:bool=False):'Each subclass must implement its own indexing logic.'