from pathlib import Path
from typing import List, Set, Tuple, Dict, Any, Optional, Iterable, overload
from functools import singledispatchmethod

from cldk.analysis.java import JavaAnalysis
from hamster.code_analysis.focal_class_method.focal_class_method import FocalClassMethod
from hamster.code_analysis.model.models import TestingFramework

from nltest.nl2test.models import (
    NL2TestInput,
    AtomicBlock,
    CandidateMethod,
    AtomicBlockList,
    LocalizedScenario,
    LocalizedStep,
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

        # Print basic results
        self._print_evaluation_results(focal_methods, optimal_coverage, coverage_score)

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

        # Print basic results
        self._print_evaluation_results(focal_methods, optimal_coverage, coverage_score)

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

        return {
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

    def _create_detailed_results_atomic(
        self,
        focal_methods: Set[Tuple[str, str]],
        covered_focal_methods: Set[Tuple[str, str]],
        coverage_score: float,
        atomic_blocks: AtomicBlockList,
    ) -> Dict[str, Any]:
        """Create detailed output for AtomicBlockList and include original blocks."""
        result = self._build_common_results(
            focal_methods, covered_focal_methods, coverage_score
        )

        block_analysis = []
        for i, block in enumerate(atomic_blocks.atomic_blocks):
            block_candidates = []
            for candidate in block.candidate_methods:
                method_key = (
                    candidate.containing_class_name,
                    candidate.method_signature,
                )
                is_covered = method_key in covered_focal_methods
                block_candidates.append(
                    {
                        "containing_class": candidate.containing_class_name,
                        "method_signature": candidate.method_signature,
                        "is_focal_method": method_key in focal_methods,
                        "is_covered": is_covered,
                    }
                )

            block_analysis.append(
                {
                    "block_index": i,
                    "simplified": block.simplified,
                    "candidate_methods": block_candidates,
                    "notes": block.notes,
                }
            )

        result.update(
            {
                "atomic_blocks_analysis": block_analysis,
                "atomic_blocks": atomic_blocks,
            }
        )
        return result

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
        result = self._build_common_results(
            focal_methods, covered_focal_methods, coverage_score
        )

        def analyze_block(block: LocalizedStep) -> Dict[str, Any]:
            cm_list = []
            for candidate in block.candidate_methods:
                method_key = (
                    candidate.containing_class_name,
                    candidate.method_signature,
                )
                cm_list.append(
                    {
                        "containing_class": candidate.containing_class_name,
                        "method_signature": candidate.method_signature,
                        "is_focal_method": method_key in focal_methods,
                        "is_covered": method_key in covered_focal_methods,
                    }
                )
            return {
                "task": getattr(block, "task", ""),
                "candidate_methods": cm_list,
                "comments": getattr(block, "comments", ""),
            }

        scenario_analysis = {
            "setup": [analyze_block(b) for b in (scenario.setup or [])],
            "steps": [
                {
                    "given": [analyze_block(b) for b in (gblock.given or [])],
                    "when": [analyze_block(b) for b in (gblock.when or [])],
                    "then": [analyze_block(b) for b in (gblock.then or [])],
                }
                for gblock in (scenario.steps or [])
            ],
            "teardown": [analyze_block(b) for b in (scenario.teardown or [])],
        }

        result.update(
            {
                "atomic_blocks_analysis": [],
                "scenario_analysis": scenario_analysis,
                "localized_scenario": scenario,
            }
        )
        return result

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

    def _print_evaluation_results(
        self,
        focal_methods: Set[Tuple[str, str]],
        covered_focal_methods: Set[Tuple[str, str]],
        coverage_score: float,
    ):
        """Print basic evaluation results."""
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

        results = {
            "test_class": self.nl2_input.qualified_class_name,
            "test_method": self.nl2_input.method_signature,
            "total_focal_methods": len(focal_methods),
            "covered_focal_methods": len(covered_focal_methods),
            "coverage_score": coverage_score,
            "focal_methods": focal_methods_str,
            "covered_methods": covered_methods_str,
            "uncovered_methods": uncovered_methods_str,
        }

        pretty_print("Localization Evaluation Results", results)
