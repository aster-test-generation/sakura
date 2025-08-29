from __future__ import annotations

from enum import Enum
from typing import List, Optional, Literal, Dict, Any, Annotated

from pydantic import BaseModel, Field, ConfigDict


class DecompositionMode(Enum):
    GHERKIN = "gherkin"
    GRAMMATICAL = "grammatical"


# ---- Gherkin Task Decomposition ----


class Block(BaseModel):
    id: int
    task: str
    uses: str
    produces: str


class GherkinBlock(BaseModel):
    given: List[Block]
    when: List[Block]
    then: List[Block]


class Scenario(BaseModel):
    testing_framework: str
    setup: List[Block]
    tasks: List[GherkinBlock]
    teardown: List[Block]


# ---- Localized Gherkin Task Decomposition ----


class CandidateMethod(BaseModel):
    implementing_class_name: str  # The class that directly implements the method
    containing_class_name: (
        str  # The class that inherits/contains the method (class under analysis)
    )
    method_signature: str
    return_type: str


class LocalizedBlock(Block):
    candidate_methods: List[CandidateMethod]
    comments: str


class LocalizedGherkinBlock(BaseModel):
    given: List[LocalizedBlock]
    when: List[LocalizedBlock]
    then: List[LocalizedBlock]


class LocalizedScenario(Scenario):
    testing_framework: str
    setup: List[LocalizedBlock]
    tasks: List[LocalizedGherkinBlock]
    teardown: List[LocalizedBlock]


class ArgBinding(BaseModel):
    arg_name: str
    arg_value: str  # Can be ${...} or a literal value


# ---- Grammatical Block Decomposition ----


class PrepPhrase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preposition: str
    object: str


class GrammaticalBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order: int  # For sequencing
    subjects: List[str]
    verbs: List[
        str
    ]  # In preprocessing, we break this down into one verb per block. We include past participle here.
    past_participles: List[str]
    direct_objs: List[str]
    indirect_objs: List[str]
    prep_phrases: List[PrepPhrase]
    polarity: Literal["positive", "negative"]
    conditions: List[str]
    simplified: str


class AtomicBlock(GrammaticalBlock):
    candidate_methods: List[CandidateMethod]
    notes: str

    @classmethod
    def from_grammatical_block(
        cls,
        gb: GrammaticalBlock,
        *,
        candidate_methods: List[CandidateMethod] | None = None,
        notes: str = "",
    ) -> "AtomicBlock":
        cm = list(candidate_methods) if candidate_methods is not None else []
        return cls(**gb.model_dump(), candidate_methods=cm, notes=notes)


class LocalizationEvaluationResults(BaseModel):
    """Detailed results from the localization grader."""

    test_class: str
    test_method: str
    total_focal_methods: int
    covered_focal_methods: int
    uncovered_focal_methods: int
    coverage_score: float
    focal_methods: List[str]
    covered_methods: List[str]
    uncovered_methods: List[str]
    atomic_blocks_analysis: List[Dict[str, Any]]
    evaluation_algorithm: str


# ---- Aggregations ----


class GrammaticalBlockList(BaseModel):
    grammatical_blocks: Annotated[
        List[GrammaticalBlock], Field(description="Ordered list of grammatical blocks.")
    ]


class AtomicBlockList(BaseModel):
    atomic_blocks: Annotated[
        List[AtomicBlock], Field(description="Ordered list of atomic blocks.")
    ]
