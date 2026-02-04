from .format_validator import FormatValidator
from .model import ClientType
from .prompt_formatting import OptimizedPrompts, PromptFormatter
from .usage_tracker import UsageTracker

__all__ = [
    "FormatValidator",
    "ClientType",
    "OptimizedPrompts",
    "PromptFormatter",
    "UsageTracker",
    "LLMClient",  # Lazily provided via __getattr__
]


def __getattr__(name: str):
    if name == "LLMClient":
        from .llm_client import LLMClient  # Local import to avoid cycles

        return LLMClient
    raise AttributeError(f"module 'sakura.utils.llm' has no attribute {name!r}")
