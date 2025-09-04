from .format_validator import FormatValidator
from .model import ClientType
from .usage_tracker import usage_tracker

__all__ = [
    "FormatValidator",
    "ClientType",
    "usage_tracker",
    "LLMClient",  # Lazily provided via __getattr__
]


def __getattr__(name: str):
    if name == "LLMClient":
        from .llm_client import LLMClient  # Local import to avoid cycles

        return LLMClient
    raise AttributeError(f"module 'nltest.utils.llm' has no attribute {name!r}")
