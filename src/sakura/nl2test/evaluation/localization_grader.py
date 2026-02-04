from __future__ import annotations

from functools import singledispatchmethod
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from cldk.analysis.java import JavaAnalysis
from hamster.code_analysis.focal_class_method.focal_class_method import (
    FocalClassMethod,
)
from hamster.code_analysis.test_statistics import (
    SetupAnalysisInfo,
)

from sakura.nl2test.models import (
    AgentState,
    AtomicBlockList,
    LocalizedScenario,
    NL2TestInput,
)
from sakura.nl2test.models.decomposition import (
    DecompositionMode,
    LocalizationEval,
    LocalizedStep,
)
from sakura.utils.analysis import CommonAnalysis
from sakura.utils.pretty.color_logger import RichLog
from sakura.utils.pretty.prints import pretty_print

# Tuple[qualified_class_name, method_sig]
FocalMethod = Tuple[str, str]
# Tuple[simple_class_name, method_name] for relaxed matching
SemanticKey = Tuple[str, str]


class LocalizationGrader:
    def __init__(
        self,
        analysis: JavaAnalysis,
        project_root: Path,
        decomposition_mode: DecompositionMode,
        application_classes: Sequence[str],
        test_utility_classes: Sequence[str] | None = None,
    ) -> None:
        self.analysis = analysis
        self.project_root = Path(project_root)
        self.decomposition_mode = decomposition_mode
        self.application_classes = list(application_classes)
        self.test_utility_classes: List[str] = list(test_utility_classes or [])
        self.common_analysis = CommonAnalysis(analysis)

    def set_analysis(self, analysis: JavaAnalysis) -> None:
        """Update the underlying analysis used for grading."""
        self.analysis = analysis
        self.common_analysis = CommonAnalysis(analysis)

    @singledispatchmethod
    def grade(self, obj, nl2_input: NL2TestInput) -> LocalizationEval:
        raise TypeError("Unsupported input type for grade().")

    @grade.register
    def _(
        self, atomic_blocks: AtomicBlockList, nl2_input: NL2TestInput
    ) -> LocalizationEval:
        raise NotImplementedError(
            "Grading for AtomicBlockList is not implemented in the new grader."
        )

    @grade.register
    def _(
        self, scenario: LocalizedScenario, nl2_input: NL2TestInput
    ) -> LocalizationEval:
        focal_methods = self._get_focal_methods(nl2_input)

        if not focal_methods:
            return self._empty_results(nl2_input)

        covered = self._collect_covered_methods(scenario, focal_methods)
        uncovered = focal_methods - covered

        tp = len(covered)
        fn = len(uncovered)
        denominator = tp + fn
        recall = tp / denominator if denominator else 1.0
        recall = round(recall, 4)

        return LocalizationEval(
            qualified_class_name=nl2_input.qualified_class_name,
            method_signature=nl2_input.method_signature,
            all_focal_methods=self._format_methods(focal_methods),
            covered_focal_methods=self._format_methods(covered),
            uncovered_focal_methods=self._format_methods(uncovered),
            tp=tp,
            fn=fn,
            localization_recall=recall,
        )

    def grade_from_state(
        self, supervisor_state: AgentState | None, nl2_input: NL2TestInput
    ) -> LocalizationEval:
        focal_methods = self._get_focal_methods(nl2_input)
        if not focal_methods:
            return self._empty_results(nl2_input)

        localization_target = None
        if supervisor_state is not None:
            if self.decomposition_mode == DecompositionMode.GHERKIN:
                localization_target = supervisor_state.localized_scenario
            else:
                localization_target = supervisor_state.atomic_blocks

        if localization_target is None:
            return self._missing_output_results(nl2_input, focal_methods)

        try:
            return self.grade(localization_target, nl2_input)
        except Exception as exc:
            RichLog.warn(
                "Localization evaluation failed for "
                f"{nl2_input.qualified_class_name}::{nl2_input.method_signature} "
                f"(id={nl2_input.id}): {exc}"
            )
            return self._missing_output_results(nl2_input, focal_methods)

    def _empty_results(self, nl2_input: NL2TestInput) -> LocalizationEval:
        return LocalizationEval(
            qualified_class_name=nl2_input.qualified_class_name,
            method_signature=nl2_input.method_signature,
            all_focal_methods=[],
            covered_focal_methods=[],
            uncovered_focal_methods=[],
            tp=0,
            fn=0,
            localization_recall=round(1.0, 4),
        )

    def _missing_output_results(
        self, nl2_input: NL2TestInput, focal_methods: Set[FocalMethod]
    ) -> LocalizationEval:
        return LocalizationEval(
            qualified_class_name=nl2_input.qualified_class_name,
            method_signature=nl2_input.method_signature,
            all_focal_methods=self._format_methods(focal_methods),
            covered_focal_methods=[],
            uncovered_focal_methods=self._format_methods(focal_methods),
            tp=0,
            fn=len(focal_methods),
            localization_recall=0.0,
        )

    def _get_focal_methods(self, nl2_input: NL2TestInput) -> Set[FocalMethod]:
        try:
            setup_methods: Dict[str, List[str]] = SetupAnalysisInfo(
                self.analysis
            ).get_setup_methods(nl2_input.qualified_class_name)

            focal_finder = FocalClassMethod(
                self.analysis, self.application_classes, self.test_utility_classes
            )
            focal_classes, _, _, _ = focal_finder.extract_test_scope(
                nl2_input.qualified_class_name,
                nl2_input.method_signature,
                setup_methods,
            )

            focal_methods: Set[FocalMethod] = set()
            for focal_class in focal_classes:
                for method_name in focal_class.focal_method_names:
                    focal_methods.add((focal_class.focal_class, method_name))

            return focal_methods
        except Exception as exc:
            pretty_print(
                "Error getting focal methods",
                {
                    "test_class": nl2_input.qualified_class_name,
                    "test_method": nl2_input.method_signature,
                    "error": str(exc),
                },
            )
            return set()

    def _get_semantic_key(
        self, qualified_class_name: str, method_signature: str
    ) -> SemanticKey:
        """
        Extract a relaxed matching key from a focal method.
        """
        simple_class = qualified_class_name.split(".")[-1].split("$")[-1]
        method_name = method_signature.split("(")[0].strip()
        return (simple_class, method_name)

    def _collect_covered_methods(
        self, scenario: LocalizedScenario, focal_methods: Set[FocalMethod]
    ) -> Set[FocalMethod]:
        """
        Identifies which ground truth focal methods were predicted by the model.
        """
        covered: Set[FocalMethod] = set()

        # Multiple overloads may map to the same key
        ground_truth_map: Dict[SemanticKey, Set[FocalMethod]] = {}
        for gt_fqn, gt_sig in focal_methods:
            key = self._get_semantic_key(gt_fqn, gt_sig)
            ground_truth_map.setdefault(key, set()).add((gt_fqn, gt_sig))

        for step in self._iter_steps(scenario):
            candidates = self._collect_candidates(step)
            for pred_fqn, pred_sig in candidates:
                pred_key = self._get_semantic_key(pred_fqn, pred_sig)
                if pred_key in ground_truth_map:
                    covered.update(ground_truth_map[pred_key])

        return covered

    def _collect_candidates(self, step: LocalizedStep) -> Set[FocalMethod]:
        results: Set[FocalMethod] = set()
        for candidate in step.candidate_methods or []:
            if candidate.containing_class_name and candidate.method_signature:
                results.add(
                    (candidate.containing_class_name, candidate.method_signature)
                )
        return results

    def _iter_steps(self, scenario: LocalizedScenario) -> Iterable[LocalizedStep]:
        yield from scenario.setup or []
        for group in scenario.gherkin_groups or []:
            yield from group.given or []
            yield from group.when or []
            yield from group.then or []
        yield from scenario.teardown or []

    def _format_methods(self, methods: Set[FocalMethod]) -> List[str]:
        return sorted(
            f"{class_name}.{method_sig}" for class_name, method_sig in methods
        )
