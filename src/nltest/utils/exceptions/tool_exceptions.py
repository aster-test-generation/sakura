from langchain_core.tools import ToolException
from typing import Dict, Any, Optional
from abc import ABC


class BaseToolException(ToolException, ABC):
    """Base class for tool-related exceptions."""
    def __init__(self, message: str, extra_info: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.extra_info = extra_info or {}

    def __str__(self) -> str:
        base_message = super().__str__()
        return f"{base_message} Details: {self.extra_info}" if self.extra_info else base_message


class InvalidArgumentError(BaseToolException):
    """Raised when an invalid argument is provided to a tool."""
    pass


class MethodNotFoundError(BaseToolException):
    """Raised when a method cannot be found in the specified class."""
    pass


class ClassNotFoundError(BaseToolException):
    """Raised when a class cannot be found."""
    pass


class CallSiteNotFoundError(BaseToolException):
    """Raised when call sites cannot be found."""
    pass


class FormatError(BaseToolException):
    """Raised when output format validation fails for LLMs."""
    pass


class ClassFileNotFound(BaseToolException):
    """Raised when a file associated to a qualified class cannot be found."""
    pass


class CompilationUnitNotFound(BaseToolException):
    """Raised when a compilation unit cannot be found."""
    pass

class AtomicBlockNotFoundError(BaseToolException):
    """Raised when an atomic block cannot be found."""
    pass