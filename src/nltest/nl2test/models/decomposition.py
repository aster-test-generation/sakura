from __future__ import annotations

from enum import Enum
import textwrap
from typing import List, Optional, Literal, Dict, Any, Annotated, Union, Tuple, Set

from pydantic import BaseModel, Field, ConfigDict

from nltest.utils.models import NL2TestInput


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
    model_config = ConfigDict(populate_by_name=True)
    given: List[Step]
    when: List[Step]
    then: List[Step]


_GHERKIN_STEPS_DESC = textwrap.dedent(
    """
    Ordered Gherkin-style step groups (given, when, then) that capture distinct behaviors or paths.
    Ids must be unique and increase across setup, each Gherkin group in order, and teardown. Each group has at least one When and one Then.
    """
).strip()


class Scenario(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    testing_framework: str
    setup: List[Step]
    gherkin_groups: Annotated[
        List[GherkinStep],
        Field(description=_GHERKIN_STEPS_DESC),
    ]
    teardown: List[Step]


# ---- Localized Gherkin Task Decomposition ----


class CandidateMethod(BaseModel):
    declaring_class_name: str  # The class that directly implements the method
    containing_class_name: (
        str  # The class that inherits/contains the method (class under analysis)
    )
    method_signature: str
    return_type: str


def _is_valid_candidate(candidate: CandidateMethod | None) -> bool:
    if candidate is None:
        return False
    return any(
        getattr(candidate, attr, "").strip()
        for attr in ("declaring_class_name", "containing_class_name", "method_signature")
    )


def _candidate_key(candidate: CandidateMethod) -> Tuple[str, str, str, str]:
    return (
        (candidate.declaring_class_name or "").strip(),
        (candidate.containing_class_name or "").strip(),
        (candidate.method_signature or "").strip(),
        (candidate.return_type or "").strip(),
    )


def _normalize_candidates(
    candidates: List[CandidateMethod],
    best_candidate: CandidateMethod | None = None,
    limit: int = 3,
) -> Tuple[List[CandidateMethod], CandidateMethod | None]:
    if limit <= 0:
        return [], best_candidate

    ordered: List[CandidateMethod] = []
    seen: Set[Tuple[str, str, str, str]] = set()

    def _add(candidate: CandidateMethod | None) -> None:
        if not _is_valid_candidate(candidate):
            return
        assert candidate is not None
        key = _candidate_key(candidate)
        if key in seen:
            return
        seen.add(key)
        ordered.append(candidate)

    _add(best_candidate)
    for c in candidates:
        _add(c)

    truncated = ordered[:limit]
    if truncated:
        return truncated, truncated[0]
    return [], best_candidate


class ArgBinding(BaseModel):
    arg_name: str
    arg_value: str  # Can be ${...} or a literal value


class LocalizedStep(Step):
    candidate_methods: List[CandidateMethod]
    arg_bindings: List[ArgBinding]
    comments: str
    external: bool

    def enforce_candidate_limit(self, limit: int = 3) -> None:
        truncated, _ = _normalize_candidates(
            self.candidate_methods, limit=limit
        )
        self.candidate_methods = truncated


class LocalizedGherkinStep(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    given: List[LocalizedStep]
    when: List[LocalizedStep]
    then: List[LocalizedStep]


class LocalizedScenario(Scenario):
    model_config = ConfigDict(populate_by_name=True)
    testing_framework: str
    setup: List[LocalizedStep]
    gherkin_groups: Annotated[
        List[LocalizedGherkinStep],
        Field(description=_GHERKIN_STEPS_DESC),
    ]
    teardown: List[LocalizedStep]

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "LocalizedScenario":
        """Create a LocalizedScenario from a plain Scenario.

        Fields not present in Scenario are initialized with empty defaults so the
        localization agent can fill them in later.
        """

        def _to_localized_step(s: Step) -> LocalizedStep:
            # Initialize required fields with empty defaults
            return LocalizedStep(
                id=s.id,
                task=s.task,
                uses=s.uses,
                produces=s.produces,
                candidate_methods=[],
                arg_bindings=[],
                comments="",
                external=False,
            )

        localized_steps: List[LocalizedGherkinStep] = []
        for gstep in scenario.gherkin_groups:
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
            gherkin_groups=localized_steps,
            teardown=[_to_localized_step(s) for s in scenario.teardown],
        )

    def enforce_candidate_limits(self, limit: int = 3) -> None:
        for step in self.setup:
            step.enforce_candidate_limit(limit)
        for grouped in self.gherkin_groups:
            for collection in (
                grouped.given,
                grouped.when,
                grouped.then,
            ):
                for step in collection:
                    step.enforce_candidate_limit(limit)
        for step in self.teardown:
            step.enforce_candidate_limit(limit)


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
    best_candidate: CandidateMethod = Field(
        default_factory=lambda: CandidateMethod(
            declaring_class_name="",
            containing_class_name="",
            method_signature="",
            return_type="",
        )
    )
    notes: str

    def enforce_candidate_limit(self, limit: int = 3) -> None:
        truncated, new_best = _normalize_candidates(
            self.candidate_methods, self.best_candidate, limit
        )
        self.candidate_methods = truncated
        if truncated:
            self.best_candidate = truncated[0]
        elif new_best is not None:
            self.best_candidate = new_best

    @classmethod
    def from_grammatical_block(
            cls,
            gb: GrammaticalBlock,
            *,
            candidate_methods: List[CandidateMethod] | None = None,
            best_candidate: CandidateMethod | None = None,
            notes: str = "",
    ) -> "AtomicBlock":
        cm = list(candidate_methods) if candidate_methods is not None else []
        bc = (
            best_candidate
            if best_candidate is not None
            else CandidateMethod(
                declaring_class_name="",
                containing_class_name="",
                method_signature="",
                return_type="",
            )
        )
        return cls(
            **gb.model_dump(), candidate_methods=cm, best_candidate=bc, notes=notes
        )


class LocalizationEvaluationResultsOld(BaseModel):
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
    evaluation_algorithm: str
    # Confusion-style metrics
    tp: int
    fp: int
    fn: int


class LocalizationEval(BaseModel):
    qualified_class_name: str  # Of ground truth
    method_signature: str
    all_focal_methods: List[str]
    covered_focal_methods: List[str]
    uncovered_focal_methods: List[str]
    tp: int  # The focal method exists in a step's candidate methods or best candidate
    fn: int  # The focal method does not exist
    localization_recall: float


# ---- Aggregations ----


class GrammaticalBlockList(BaseModel):
    grammatical_blocks: Annotated[
        List[GrammaticalBlock], Field(description="Ordered list of grammatical blocks.")
    ]


class AtomicBlockList(BaseModel):
    atomic_blocks: Annotated[
        List[AtomicBlock], Field(description="Ordered list of atomic blocks.")
    ]

    def enforce_candidate_limits(self, limit: int = 3) -> None:
        for block in self.atomic_blocks:
            block.enforce_candidate_limit(limit)


class NL2LocalizationOutput(BaseModel):
    """Combined output from NL2Test localization evaluation."""

    nl2_input: NL2TestInput
    localized_blocks: Union[AtomicBlockList, LocalizedScenario]
    evaluation_results: Optional[
        Union[LocalizationEvaluationResultsOld, LocalizationEval]
    ] = None
