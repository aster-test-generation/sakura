from pathlib import Path
from typing import Set, Tuple, Dict, Any, Optional, Iterable, overload
from functools import singledispatchmethod

from cldk.analysis.java import JavaAnalysis
from hamster.code_analysis.focal_class_method.focal_class_method import FocalClassMethod

from nltest.nl2test.models import (
    NL2TestInput,
    AtomicBlock,
    AtomicBlockList,
    LocalizedScenario,
)
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.utils.analysis import CommonAnalysis
from nltest.utils.pretty.prints import pretty_print


class LocalizationGrader:
    @overload
    def grade(
        self, atomic_blocks: AtomicBlockList, detailed_output: bool = False
    ) -> Tuple[float, Optional[Dict[str, Any]]]: ...

    @overload
    def grade(
        self, scenario: LocalizedScenario, detailed_output: bool = False
    ) -> Tuple[float, Optional[Dict[str, Any]]]: ...

    def __init__(
        self,
        nl2_input: NL2TestInput,
        analysis: JavaAnalysis,
        project_root: Path,
        *,
        decomposition_mode: DecompositionMode = DecompositionMode.GRAMMATICAL,
    ):
        self.nl2_input = nl2_input
        self.analysis = analysis
        self.project_root = project_root
        self.common_analysis = CommonAnalysis(analysis)
        self.decomposition_mode = decomposition_mode

    @singledispatchmethod
    def grade(
        self, _, detailed_output: bool = False
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """Dispatch based on input type (AtomicBlockList or LocalizedScenario)."""
        raise TypeError("Unsupported input type for grade().")

    @grade.register
    def _(
        self, atomic_blocks: AtomicBlockList, detailed_output: bool = False
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        Grade the localization of atomic blocks to focal methods.

        Args:
            atomic_blocks: List of atomic blocks with candidate methods
            detailed_output: If True, returns comprehensive evaluation results

        Returns:
            Tuple[float, Optional[Dict[str, Any]]]:
                - Percentage of focal methods covered (0.0 to 1.0)
                - Detailed results dict if detailed_output=True, None otherwise
        """
        focal_methods = self._get_focal_methods()

        if not focal_methods:
            pretty_print(
                "No focal methods found",
                {
                    "test_class": self.nl2_input.qualified_class_name,
                    "test_method": self.nl2_input.method_signature,
                },
            )
            return 0.0, (
                None
                if not detailed_output
                else self._create_detailed_results_atomic(
                    set(), set(), 0.0, atomic_blocks
                )
            )

        # Compute optimal coverage generically over candidate-bearing blocks
        optimal_coverage = self._find_optimal_coverage(
            focal_methods, self._iter_candidate_blocks(atomic_blocks)
        )

        # Calculate score
        coverage_score = (
            len(optimal_coverage) / len(focal_methods) if focal_methods else 0.0
        )

        # No early printing; detailed metrics are computed only for detailed output

        # Return detailed results if requested
        detailed_results = None
        if detailed_output:
            detailed_results = self._create_detailed_results_atomic(
                focal_methods, optimal_coverage, coverage_score, atomic_blocks
            )

        return coverage_score, detailed_results

    @grade.register
    def _(
        self, blocks: list, detailed_output: bool = False
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """Compatibility: accept List[AtomicBlock] by wrapping into AtomicBlockList."""
        if all(isinstance(b, AtomicBlock) for b in blocks):
            return self.grade(
                AtomicBlockList(atomic_blocks=list(blocks)), detailed_output
            )
        raise TypeError(
            "Unsupported list contents for grade(); expected List[AtomicBlock]."
        )

    @grade.register
    def _(
        self, scenario: LocalizedScenario, detailed_output: bool = False
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        Grade the localization for a LocalizedScenario by flattening its blocks and
        applying the same coverage algorithm used for atomic blocks.
        """
        focal_methods = self._get_focal_methods()

        if not focal_methods:
            pretty_print(
                "No focal methods found",
                {
                    "test_class": self.nl2_input.qualified_class_name,
                    "test_method": self.nl2_input.method_signature,
                },
            )
            return 0.0, (
                None
                if not detailed_output
                else self._create_detailed_results_scenario(set(), set(), 0.0, scenario)
            )

        # Compute optimal coverage generically over candidate-bearing blocks
        optimal_coverage = self._find_optimal_coverage(
            focal_methods, self._iter_candidate_blocks(scenario)
        )

        # Calculate score
        coverage_score = (
            len(optimal_coverage) / len(focal_methods) if focal_methods else 0.0
        )

        # No early printing; detailed metrics are computed only for detailed output

        detailed_results = None
        if detailed_output:
            detailed_results = self._create_detailed_results_scenario(
                focal_methods, optimal_coverage, coverage_score, scenario
            )

        return coverage_score, detailed_results

    def _find_optimal_coverage(
        self, focal_methods: Set[Tuple[str, str]], blocks: Iterable[Any]
    ) -> Set[Tuple[str, str]]:
        """Generic one-pass coverage finder over any blocks with candidate_methods.

        Each focal method can be matched at most once, across all blocks in order.
        """
        covered_focal_methods: Set[Tuple[str, str]] = set()
        available_focal_methods = set(focal_methods)

        for block in blocks:
            block_candidates: Set[Tuple[str, str]] = set()

            for cm in getattr(block, "candidate_methods", []) or []:
                block_candidates.add((cm.containing_class_name, cm.method_signature))

            best = getattr(block, "best_candidate", None)
            if best is not None:
                # Guard against placeholder/empty candidates
                if getattr(best, "containing_class_name", "") and getattr(
                    best, "method_signature", ""
                ):
                    block_candidates.add(
                        (best.containing_class_name, best.method_signature)
                    )

            for focal_method in list(available_focal_methods):
                if focal_method in block_candidates:
                    covered_focal_methods.add(focal_method)
                    available_focal_methods.remove(focal_method)
                    break

        return covered_focal_methods

    def _build_common_results(
        self,
        focal_methods: Set[Tuple[str, str]],
        covered_focal_methods: Set[Tuple[str, str]],
        coverage_score: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        focal_methods_str = [
            f"{class_name}.{method_sig}" for class_name, method_sig in focal_methods
        ]
        covered_methods_str = [
            f"{class_name}.{method_sig}"
            for class_name, method_sig in covered_focal_methods
        ]
        uncovered_methods_str = [
            f"{class_name}.{method_sig}"
            for class_name, method_sig in (focal_methods - covered_focal_methods)
        ]

        base: Dict[str, Any] = {
            "test_class": self.nl2_input.qualified_class_name,
            "test_method": self.nl2_input.method_signature,
            "total_focal_methods": len(focal_methods),
            "covered_focal_methods": len(covered_focal_methods),
            "uncovered_focal_methods": len(uncovered_methods_str),
            "coverage_score": coverage_score,
            "focal_methods": focal_methods_str,
            "covered_methods": covered_methods_str,
            "uncovered_methods": uncovered_methods_str,
            "evaluation_algorithm": "optimal_coverage_one_to_one",
        }
        if extra:
            base.update(extra)
        return base

    def _create_detailed_results_atomic(
        self,
        focal_methods: Set[Tuple[str, str]],
        covered_focal_methods: Set[Tuple[str, str]],
        coverage_score: float,
        atomic_blocks: AtomicBlockList,
    ) -> Dict[str, Any]:
        """Detailed output: common metrics + provided atomic blocks only."""
        # Include TP/FP/FN with block-level FP definition for these blocks
        extra = self._compute_confusion_counts(
            focal_methods, covered_focal_methods, atomic_blocks
        )
        return self._build_common_results(
            focal_methods, covered_focal_methods, coverage_score, extra
        )

    def _create_detailed_results_scenario(
        self,
        focal_methods: Set[Tuple[str, str]],
        covered_focal_methods: Set[Tuple[str, str]],
        coverage_score: float,
        scenario: LocalizedScenario,
    ) -> Dict[str, Any]:
        """Create detailed output for LocalizedScenario and include original structure.

        Does not flatten the scenario in the output.
        """
        extra = self._compute_confusion_counts(
            focal_methods, covered_focal_methods, scenario
        )
        return self._build_common_results(
            focal_methods, covered_focal_methods, coverage_score, extra
        )

    def _iter_candidate_blocks(
        self, obj: AtomicBlockList | LocalizedScenario
    ) -> Iterable[Any]:
        """Yield blocks with candidate_methods in evaluation order for either input type.

        - AtomicBlockList: yields in atomic block order
        - LocalizedScenario: yields setup, then each gherkin block's given/when/then, then teardown
        """
        if isinstance(obj, AtomicBlockList):
            yield from (obj.atomic_blocks or [])
            return
        if isinstance(obj, LocalizedScenario):
            # Setup
            for b in obj.setup or []:
                yield b
            # Steps (gherkin groups)
            for gblock in obj.steps or []:
                for b in gblock.given or []:
                    yield b
                for b in gblock.when or []:
                    yield b
                for b in gblock.then or []:
                    yield b
            # Teardown
            for b in obj.teardown or []:
                yield b
            return
        # Should never reach here due to typing/dispatch
        raise TypeError("Unsupported object type for iteration")

    def _collect_identified_methods(
        self, obj: AtomicBlockList | LocalizedScenario
    ) -> Set[Tuple[str, str]]:
        """Collect all methods identified across blocks (candidates and best).

        Returns a set of (class_name, method_signature) tuples.
        """
        identified: Set[Tuple[str, str]] = set()
        for block in self._iter_candidate_blocks(obj):
            for cm in getattr(block, "candidate_methods", []) or []:
                identified.add((cm.containing_class_name, cm.method_signature))

            best = getattr(block, "best_candidate", None)
            if best is not None:
                if getattr(best, "containing_class_name", "") and getattr(
                    best, "method_signature", ""
                ):
                    identified.add((best.containing_class_name, best.method_signature))
        return identified

    def _compute_confusion_counts(
        self,
        focal_methods: Set[Tuple[str, str]],
        covered_focal_methods: Set[Tuple[str, str]],
        obj: AtomicBlockList | LocalizedScenario,
    ) -> Dict[str, int]:
        """Compute TP/FP/FN counts with block-level false positives.

        - TP: number of focal methods covered (i.e., matched in optimal coverage).
        - FP: number of blocks that have at least one candidate (candidate_methods and/or
              best_candidate) but none of those candidates is a focal method.
        - FN: number of focal methods that were not identified in any block.
        """
        # True positives are based on coverage
        tp = len(covered_focal_methods)

        # False positives are counted per block: non-empty candidates but none are focal
        fp_blocks = 0
        for block in self._iter_candidate_blocks(obj):
            block_candidates: Set[Tuple[str, str]] = set()

            for cm in getattr(block, "candidate_methods", []) or []:
                block_candidates.add((cm.containing_class_name, cm.method_signature))

            best = getattr(block, "best_candidate", None)
            if best is not None:
                if getattr(best, "containing_class_name", "") and getattr(
                    best, "method_signature", ""
                ):
                    block_candidates.add(
                        (best.containing_class_name, best.method_signature)
                    )

            if block_candidates and not any(c in focal_methods for c in block_candidates):
                fp_blocks += 1

        # False negatives relative to all identified methods across blocks
        identified_methods = self._collect_identified_methods(obj)
        fn = len(focal_methods - identified_methods)

        return {"tp": tp, "fp": fp_blocks, "fn": fn}

    def _get_focal_methods(self) -> Set[Tuple[str, str]]:
        """Use Hamster to get the focal methods for the test method."""
        try:
            testing_frameworks = self.common_analysis.get_testing_frameworks_for_class(
                self.nl2_input.qualified_class_name
            )
            setup_methods = self.common_analysis.get_setup_methods(
                self.nl2_input.qualified_class_name
            )
            setup_method_signatures = [method.signature for method in setup_methods]

            _, application_classes = (
                self.common_analysis.get_test_methods_classes_and_application_classes()
            )

            focal_class_method = FocalClassMethod(
                self.analysis, testing_frameworks, application_classes
            )
            focal_classes, _, _, _ = (
                focal_class_method.identify_focal_class_and_ui_api_test(
                    self.nl2_input.qualified_class_name,
                    self.nl2_input.method_signature,
                    setup_method_signatures,
                )
            )

            focal_methods = set()
            for focal_class in focal_classes:
                for method_name in focal_class.focal_method_names:
                    focal_methods.add((focal_class.focal_class, method_name))

            return focal_methods

        except Exception as e:
            pretty_print("Error getting focal methods", {"error": str(e)})
            return set()

    # Removed printing method to avoid side effects during evaluation
    # def _print_evaluation_results(...): pass
