from __future__ import annotations

from enum import Enum
from pydantic import BaseModel


class SnippetType(Enum):
    CLASS = "class"
    METHOD = "method"


class Snippet(BaseModel):
    declaring_class_name: str  # The class that directly implements the method


class MethodSnippet(Snippet):
    code: str
    method_signature: str
    containing_class_name: (
        str  # The class that inherits/contains the method (class under analysis)
    )


class ClassSnippet(Snippet):
    simple_class_name: str

