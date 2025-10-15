from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


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


class AgentToolLog(BaseModel):
    tool_counts: Dict[str, int]
    tool_trajectories: List[List[str]]


class ToolLog(BaseModel):
    supervisor_tool_log: AgentToolLog
    localization_tool_log: AgentToolLog
    composition_tool_log: AgentToolLog


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


class NL2TestMetadata(BaseModel):
    qualified_test_class_name: str
    code: str
    # Optional predicted test method signature (e.g., testFindAll())
    method_signature: Optional[str] = None


class NL2TestStructuralEval(BaseModel):
    """Structural precision/recall metrics for the generated test."""

    obj_creation_recall: float
    obj_creation_precision: float
    assertion_recall: float
    assertion_precision: float
    callable_recall: float
    callable_precision: float
    focal_recall: float
    focal_precision: float


class NL2TestCoverageEval(BaseModel):
    class_coverage: float
    method_coverage: float
    line_coverage: float
    branch_coverage: float


class NL2TestEval(BaseModel):
    compiles: bool
    nl2test_input: NL2TestInput
    nl2test_metadata: NL2TestMetadata
    structured_eval: Optional[NL2TestStructuralEval]
    coverage_eval: Optional[NL2TestCoverageEval]
    localization_eval: Optional[Any] = None  # Accepts LocalizationEval from nl2test flows.
    tool_log: Optional[ToolLog] = None
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
