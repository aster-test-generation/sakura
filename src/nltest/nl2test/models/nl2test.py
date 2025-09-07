from __future__ import annotations

from enum import Enum
from typing import Annotated, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict

from .decomposition import (
    AtomicBlock,
    AtomicBlockList,
    LocalizationEvaluationResults,
    LocalizedScenario,
)


class AbstractionLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class NL2TestInput(BaseModel):
    id: int = -1
    description: str
    project_name: str
    qualified_class_name: Annotated[
        str, "The qualified class name of the class containing the test"
    ]
    method_signature: Annotated[str, "The method signature of the test case"]
    abstraction_level: AbstractionLevel | None = None
    is_bdd: bool = False

    model_config = ConfigDict(use_enum_values=True)


class NL2LocalizationOutput(BaseModel):
    """Combined output from NL2Test localization evaluation."""

    nl2_input: NL2TestInput
    localized_blocks: Union[AtomicBlockList, LocalizedScenario]
    evaluation_results: Optional[LocalizationEvaluationResults] = None
