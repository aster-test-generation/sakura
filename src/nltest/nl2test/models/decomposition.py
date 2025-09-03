from __future__ import annotations

from enum import Enum
from typing import List, Optional, Literal, Dict, Any, Annotated

from pydantic import BaseModel, Field, ConfigDict


class DecompositionMode(Enum):
    GHERKIN = "gherkin"
    GRAMMATICAL = "grammatical"


# ---- Gherkin Task Decomposition ----


class Step(BaseModel):
    id: int
    task: str
    uses: str
    produces: str


class GherkinStep(BaseModel):
    given: List[Step]
    when: List[Step]
    then: List[Step]


class Scenario(BaseModel):
    testing_framework: str
    setup: List[Step]
    steps: List[GherkinStep]
    teardown: List[Step]


# ---- Localized Gherkin Task Decomposition ----


class CandidateMethod(BaseModel):
    implementing_class_name: str  # The class that directly implements the method
    containing_class_name: (
        str  # The class that inherits/contains the method (class under analysis)
    )
    method_signature: str
    return_type: str


class LocalizedStep(Step):
    candidate_methods: List[CandidateMethod]
    best_candidate: CandidateMethod
    arg_bindings: List[ArgBinding]
    comments: str


class LocalizedGherkinStep(BaseModel):
    given: List[LocalizedStep]
    when: List[LocalizedStep]
    then: List[LocalizedStep]


class LocalizedScenario(Scenario):
    testing_framework: str
    setup: List[LocalizedStep]
    steps: List[LocalizedGherkinStep]
    teardown: List[LocalizedStep]

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "LocalizedScenario":
        """Create a LocalizedScenario from a plain Scenario.

        Fields not present in Scenario are initialized with empty defaults so the
        localization agent can fill them in later.
        """

        def _to_localized_step(s: Step) -> LocalizedStep:
            # Initialize required fields with empty defaults
            empty_candidate = CandidateMethod(
                implementing_class_name="",
                containing_class_name="",
                method_signature="",
                return_type="",
            )
            return LocalizedStep(
                id=s.id,
                task=s.task,
                uses=s.uses,
                produces=s.produces,
                candidate_methods=[],
                best_candidate=empty_candidate,
                arg_bindings=[],
                comments="",
            )

        localized_steps: List[LocalizedGherkinStep] = []
        for gstep in scenario.steps:
            localized_steps.append(
                LocalizedGherkinStep(
                    given=[_to_localized_step(s) for s in gstep.given],
                    when=[_to_localized_step(s) for s in gstep.when],
                    then=[_to_localized_step(s) for s in gstep.then],
                )
            )

        return cls(
            testing_framework=scenario.testing_framework,
            setup=[_to_localized_step(s) for s in scenario.setup],
            steps=localized_steps,
            teardown=[_to_localized_step(s) for s in scenario.teardown],
        )


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
    best_candidate: CandidateMethod
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
    # Include original structures for analysis without flattening
    atomic_blocks: Optional[AtomicBlockList] = None
    localized_scenario: Optional[LocalizedScenario] = None
    scenario_analysis: Optional[Dict[str, Any]] = None


# ---- Aggregations ----


class GrammaticalBlockList(BaseModel):
    grammatical_blocks: Annotated[
        List[GrammaticalBlock], Field(description="Ordered list of grammatical blocks.")
    ]


class AtomicBlockList(BaseModel):
    atomic_blocks: Annotated[
        List[AtomicBlock], Field(description="Ordered list of atomic blocks.")
    ]
