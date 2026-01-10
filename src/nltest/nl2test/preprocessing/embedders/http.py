from typing import List, Optional

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from .base import BaseEmbedder


class HttpEmbedder(BaseEmbedder):
    def __init__(self, model_id: str, api_url: str, api_key: Optional[str] = None):
        self._client = OpenAIEmbeddings(
            model=model_id,
            base_url=api_url.rstrip("/"),
            api_key=SecretStr(api_key) if api_key else None,
            check_embedding_ctx_length=False,
        )
        probe_embedding = self._client.embed_query("probe")
        super().__init__(len(probe_embedding))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)
