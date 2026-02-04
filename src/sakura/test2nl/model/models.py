from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel

from sakura.utils.constants import ABSTRACTION_TEMPERATURES


class AbstractionLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    def get_temperature(self) -> float:
        """Return the temperature for this abstraction level."""
        return ABSTRACTION_TEMPERATURES[self.value]


class TrialType(Enum):
    DESCRIPTION = "description"
    TEST_CASE = "test_case"


class TestDescriptionInfo(BaseModel):
    description: str
    prompt: str
    abstraction_level: AbstractionLevel
    temperature: float
    method_signature: str
    qualified_class_name: str
    trial_number: int = 1
    id: int = -1  # For matching Test2NL dataset with generated description


class RoundTripTest(BaseModel):
    prompt: str
    temperature: float
    generated_test: str
    method_signature: str
    qualified_class_name: str
    generated_description: TestDescriptionInfo
    score: float | None = None


class ReferencedClasses(BaseModel):
    """DEPRECATED"""

    referenced_classes: List[ClassContext]


class ClassContext(BaseModel):
    simple_class_name: str
    qualified_class_name: str
    annotations: List[str] | None = None
    extends: List[str] | None = None
    modifiers: List[str] | None = None
    field_declarations: List[FieldDeclaration] | None = None
    relevant_class_methods: List[MethodContext] | None = None
    javadoc: List[str] | None = None


class FieldDeclaration(BaseModel):
    variables: List[str] | None = None
    type: str | None = None
    modifiers: List[str] | None = None
    annotations: List[str] | None = None
    type_is_helper_class: bool | None = None


class CallSiteInfo(BaseModel):
    method_name: str
    receiver_type: str
    return_type: str
    line_number: int
    is_assertion: bool
    is_helper: bool | None = None


class VariableInfo(BaseModel):
    name: str
    type: str
    initializer: str | None = None
    line_number: int
    type_is_helper_class: bool | None = None


class MethodContext(BaseModel):
    method_signature: str
    qualified_class_name: str | None = None
    is_getter_or_setter: bool | None = None
    code: str | None = None
    call_sites: List[CallSiteInfo] = []
    variable_declarations: List[VariableInfo] = []
    thrown_exceptions: List[str] = []
    javadoc: str | None = None


class Test2NLEntry(BaseModel):
    id: int = -1
    description: str
    project_name: str
    qualified_class_name: str
    method_signature: str
    abstraction_level: AbstractionLevel | None = None
    is_bdd: bool = False

    @classmethod
    def from_test_description_info(
        cls, test_description_info: TestDescriptionInfo, project_name: str
    ) -> Test2NLEntry:
        return cls(
            id=test_description_info.id,
            description=test_description_info.description,
            project_name=project_name,
            qualified_class_name=test_description_info.qualified_class_name,
            method_signature=test_description_info.method_signature,
            abstraction_level=test_description_info.abstraction_level,
            is_bdd=False,
        )
