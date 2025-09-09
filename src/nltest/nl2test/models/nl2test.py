from __future__ import annotations

from enum import Enum
from typing import Annotated, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict

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


class NL2EvaluationResults(BaseModel):
    """Evaluation results for a single NL2Test generation run."""

    nl2_input: NL2TestInput

    # Prediction identifiers
    pred_class_name: str = ""
    pred_method_signature: str = ""

    # Ground truth identifiers
    gt_class_name: str = ""
    gt_method_signature: str = ""

    # Structural grading
    structural_score: float = 0.0
    structural_metrics: Dict[str, float] = Field(default_factory=dict)

    # Snapshot of generated code
    test_code: str = ""
