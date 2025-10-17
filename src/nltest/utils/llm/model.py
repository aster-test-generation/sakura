from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any


class Provider(Enum):
    OPENROUTER = "openrouter"
    VLLM = "vllm"
    OLLAMA = "ollama"
    OPENAI = "openai"
    GCP = "gcp"


@dataclass
class LLMSettings:
    provider: Provider
    model: str
    client_type: Optional["ClientType"] = None
    temperature: float = -1
    max_tokens: Optional[int] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    request_timeout: int = 120
    # Extra HTTP headers (e.g., OpenRouter analytics)
    default_headers: Dict[str, str] = field(default_factory=dict)
    # Arbitrary model-specific kwargs (passed through to API)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


class ClientType(Enum):
    CODE_GEN = "code_gen"
    SUMMARIZATION = "summarization"
    DECISION = "decision"
    STRUCTURED = "structured"
