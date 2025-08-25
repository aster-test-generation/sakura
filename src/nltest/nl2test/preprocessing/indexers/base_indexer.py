from abc import ABC, abstractmethod

from cldk.analysis.java import JavaAnalysis

from nltest.nl2test.preprocessing.embedders import HttpEmbedder, OllamaEmbedder
from nltest.utils.config import Config
from nltest.utils.llm.model import Provider


class BaseIndexer(ABC):
    def __init__(self, analysis: JavaAnalysis):
        self.analysis = analysis
        self.config = Config()
        self.embedder = self._initialize_embedder()

    def _initialize_embedder(self):
        raw_provider = self.config.get("emb", "provider")
        emb_model = self.config.get("emb", "model")

        try:
            provider = Provider(raw_provider)
        except ValueError:
            raise ValueError(f"Invalid embedding provider: {raw_provider}")

        if provider == Provider.VLLM:
            api_url = self.config.get(provider, "emb_api_url")
            if not api_url:
                raise ValueError("api_url missing in config for Vela provider")
            return HttpEmbedder(model_id=emb_model, api_url=api_url)

        elif provider == Provider.OLLAMA:
            return OllamaEmbedder(model_id=emb_model)

        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    @abstractmethod
    def build_index(self):
        """Each subclass must implement its own indexing logic."""
        pass
