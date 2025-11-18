from __future__ import annotations

from functools import singledispatchmethod
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple, Dict

from cldk.analysis.java import JavaAnalysis
from hamster.code_analysis.focal_class_method.focal_class_method import (
    FocalClassMethod,
)
from hamster.code_analysis.test_statistics import (
    SetupAnalysisInfo,
)

from nltest.nl2test.models import AtomicBlockList, LocalizedScenario, NL2TestInput
from nltest.nl2test.models.decomposition import (
    CandidateMethod,
    DecompositionMode,
    LocalizationEval,
    LocalizedStep,
)
from nltest.utils.analysis import CommonAnalysis
from nltest.utils.pretty.prints import pretty_print

# Tuple[qualified_class_name, method_sig]
FocalMethod = Tuple[str, str]


class LocalizationGrader:
    def __init__(
        self,
        analysis: JavaAnalysis,
        project_root: Path,
        decomposition_mode: DecompositionMode,
        application_classes: Sequence[str],
    ) -> None:
        self.analysis = analysis
        self.project_root = Path(project_root)
        self.decomposition_mode = decomposition_mode
        self.application_classes = list(application_classes)
        self.common_analysis = CommonAnalysis(analysis)

    def set_analysis(self, analysis: JavaAnalysis) -> None:
        """Update the underlying analysis used for grading."""
        self.analysis = analysis
        self.common_analysis = CommonAnalysis(analysis)

    @singledispatchmethod
    def grade(
        self, obj, nl2_input: NL2TestInput
    ) -> LocalizationEval:  
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

    def _get_focal_methods(self, nl2_input: NL2TestInput) -> Set[FocalMethod]:
        try:
            testing_frameworks = self.common_analysis.get_testing_frameworks_for_class(
                nl2_input.qualified_class_name
            )
            setup_methods: Dict[str, List[str]] = SetupAnalysisInfo(self.analysis).get_setup_methods(
                nl2_input.qualified_class_name
            )

            focal_finder = FocalClassMethod(
                self.analysis, self.application_classes
            )
            focal_classes, _, _, _ = focal_finder.identify_focal_class_and_ui_api_test(
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

    def _collect_covered_methods(
        self, scenario: LocalizedScenario, focal_methods: Set[FocalMethod]
    ) -> Set[FocalMethod]:
        covered: Set[FocalMethod] = set()
        for step in self._iter_steps(scenario):
            candidates = self._collect_candidates(step)
            for candidate in candidates:
                if candidate in focal_methods:
                    covered.add(candidate)
        return covered

    def _collect_candidates(self, step: LocalizedStep) -> Set[FocalMethod]:
        results: Set[FocalMethod] = set()
        for candidate in step.candidate_methods or []:
            self._maybe_add_candidate(results, candidate)
        # self._maybe_add_candidate(results, step.best_candidate)
        return results

    def _maybe_add_candidate(
        self, results: Set[FocalMethod], candidate: CandidateMethod | None
    ) -> None:
        if candidate is None:
            return
        containing_class = getattr(candidate, "containing_class_name", "")
        method_signature = getattr(candidate, "method_signature", "")
        if containing_class and method_signature:
            results.add((containing_class, method_signature))

    def _iter_steps(self, scenario: LocalizedScenario) -> Iterable[LocalizedStep]:
        yield from scenario.setup or []
        for group in scenario.steps or []:
            yield from group.given or []
            yield from group.when or []
            yield from group.then or []
        yield from scenario.teardown or []

    def _format_methods(self, methods: Set[FocalMethod]) -> List[str]:
        return sorted(
            f"{class_name}.{method_sig}" for class_name, method_sig in methods
        )
