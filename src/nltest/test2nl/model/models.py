from __future__ import annotations

from enum import Enum
from typing import Optional, List

from pydantic import BaseModel


class AbstractionLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TrialType(Enum):
    DESCRIPTION = "description"
    TEST_CASE = "test_case"


class GeneratedDescription(BaseModel):
    description: str
    prompt: str
    abstraction_level: AbstractionLevel
    temperature: float
    method_signature: str
    qualified_class_name: str
    trial_number: int = 1
    entry_id: int = -1 # For matching Test2NL dataset with generated description


class RoundTripTest(BaseModel):
    prompt: str
    temperature: float
    generated_test: str
    method_signature: str
    qualified_class_name: str
    generated_description: GeneratedDescription
    score: float | None = None


class ReferencedClasses(BaseModel):
    """DEPRECATED"""
    referenced_classes: List[ReferencedClass]


class ReferencedClass(BaseModel):
    simple_class_name: str
    qualified_class_name: str
    annotations: List[str] | None = None
    extends: List[str] | None = None
    modifiers: List[str] | None = None
    field_declarations: List[FieldDeclaration] | None = None
    class_methods: List[MethodContext] | None = None
    comments_in_class: List[str] | None = None  # Only Javadoc


class FieldDeclaration(BaseModel):
    variables: List[str]
    type: str | None = None
    modifiers: List[str] | None = None
    annotations: List[str] | None = None


class MethodContext(BaseModel):
    method_signature: str
    is_getter_or_setter: bool | None = None
    code: str | None = None


class Test2NLEntry(BaseModel):
    id: int = -1
    description: str
    project_name: str
    qualified_class_name: str
    method_signature: str
    abstraction_level: AbstractionLevel | None = None
    is_bdd: bool = False