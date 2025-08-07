from enum import Enum


class ProviderType(str, Enum):
    """Defines the types of providers that can be used for LLM call"""

    bam = "BAM"
    rits = "RITS"
    vela = "VELA"
    openai = "OPENAI"
    replicate = "replicate"
    ollama = "ollama"
    azure = "AZURE"


class DecodingType(str, Enum):
    """Defines the types of providers that can be used for LLM call"""

    greedy = "greedy"
    sample = "sample"
