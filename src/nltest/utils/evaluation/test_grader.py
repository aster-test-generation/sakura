from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from cldk.analysis.java import JavaAnalysis
from cldk.models.java import JCallable
from hamster.code_analysis.focal_class_method.focal_class_method import (
    FocalClassMethod,
)
from hamster.code_analysis.model.models import (
    AssertionDetails,
    AssertionType,
    CallAndAssertionSequenceDetails,
    CallableDetails,
)
from hamster.code_analysis.test_statistics import (
    SetupAnalysisInfo,
    TestMethodAnalysisInfo,
)

from nltest.utils.models import (
    NL2TestCoverageEval,
    NL2TestInput,
    NL2TestMetadata,
    NL2TestStructuralEval,
)
from nltest.utils.analysis import CommonAnalysis
from nltest.utils.coverage.individual_test_coverage import IndividualTestCoverage
from nltest.utils.pretty.color_logger import RichLog


class TestGrader:
    """
    Grader for NL2Test structural and coverage evaluation.
    """

    def __init__(
        self,
        analysis: JavaAnalysis,
        project_root: Path,
        application_classes: Optional[List[str]] = None,
        test_utility_classes: Optional[List[str]] = None,
    ) -> None:
        self.analysis = analysis
        self.project_root = project_root
        self.application_classes: List[str] = list(application_classes or [])
        self.test_utility_classes: List[str] = list(test_utility_classes or [])
        self.common = CommonAnalysis(analysis)

    def set_analysis(self, analysis: JavaAnalysis) -> None:
        self.analysis = analysis
        self.common = CommonAnalysis(analysis)

    @staticmethod
    def _count_num_assertions(seqs: List[CallAndAssertionSequenceDetails]) -> int:
        return sum(len(seq.assertion_details) for seq in seqs)

    @staticmethod
    def _get_assertion_types(
        seqs: List[CallAndAssertionSequenceDetails],
    ) -> List[List[AssertionType]]:
        assertion_types: List[List[AssertionType]] = []
        for seq in seqs:
            for assertion in seq.assertion_details:
                assertion_types.append(assertion.assertion_type)
        return assertion_types

    @staticmethod
    def _get_expanded_seqs(
        seqs: List[CallAndAssertionSequenceDetails],
    ) -> List[CallableDetails | AssertionDetails]:
        expanded: List[CallableDetails | AssertionDetails] = []
        for seq in seqs:
            expanded.extend(seq.call_sequence_details)
            expanded.extend(seq.assertion_details)
        return expanded

    @staticmethod
    def _assertion_scores(
        gt_assertions: List[List[AssertionType]],
        pred_assertions: List[List[AssertionType]],
    ) -> Tuple[float, float]:
        """Compute recall and precision over assertion groups."""
        gt_groups = [set(group or []) for group in gt_assertions]
        pred_groups = [set(group or []) for group in pred_assertions]

        matched_gt_indices: Set[int] = set()
        matched_pred_indices: Set[int] = set()

        for gt_idx, gt_group in enumerate(gt_groups):
            for pred_idx, pred_group in enumerate(pred_groups):
                if pred_idx in matched_pred_indices:
                    continue
                if any(assertion in pred_group for assertion in gt_group):
                    matched_gt_indices.add(gt_idx)
                    matched_pred_indices.add(pred_idx)
                    break

        recall = len(matched_gt_indices) / len(gt_groups) if gt_groups else 1.0
        precision = len(matched_pred_indices) / len(pred_groups) if pred_groups else 1.0
        return recall, precision

    @staticmethod
    def _callable_scores(
        gt_seqs: List[CallAndAssertionSequenceDetails],
        pred_seqs: List[CallAndAssertionSequenceDetails],
    ) -> Tuple[float, float]:
        """Compute recall and precision for callables and assertions combined."""
        from collections import Counter

        gt_expanded = TestGrader._get_expanded_seqs(gt_seqs)
        pred_expanded = TestGrader._get_expanded_seqs(pred_seqs)

        # Maybe include receiver type in matching
        gt_calls = Counter(
            (seq.receiver_type, seq.method_name)
            for seq in gt_expanded
            if isinstance(seq, CallableDetails)
        )
        pred_calls = Counter(
            (seq.receiver_type, seq.method_name)
            for seq in pred_expanded
            if isinstance(seq, CallableDetails)
        )

        gt_assert_details = [
            detail for detail in gt_expanded if isinstance(detail, AssertionDetails)
        ]
        pred_assert_details = [
            detail for detail in pred_expanded if isinstance(detail, AssertionDetails)
        ]

        gt_assert_types = [
            set(detail.assertion_type or []) for detail in gt_assert_details
        ]
        pred_assert_types = [
            set(detail.assertion_type or []) for detail in pred_assert_details
        ]

        # Count matched assertion groups by compatible assertion types
        matched_asserts = 0
        used_pred_indices: Set[int] = set()
        for gt_types in gt_assert_types:
            for idx, pred_types in enumerate(pred_assert_types):
                if idx in used_pred_indices:
                    continue
                if any(a in pred_types for a in gt_types):
                    matched_asserts += 1
                    used_pred_indices.add(idx)
                    break

        matched_calls = (gt_calls & pred_calls).total()
        total_gt = sum(gt_calls.values()) + len(gt_assert_types)
        total_pred = sum(pred_calls.values()) + len(pred_assert_types)

        total_matched = matched_calls + matched_asserts

        recall = total_matched / total_gt if total_gt else 1.0
        precision = total_matched / total_pred if total_pred else 1.0
        return recall, precision

    @staticmethod
    def _zero_structural_eval() -> NL2TestStructuralEval:
        return NL2TestStructuralEval(
            obj_creation_recall=0.0,
            obj_creation_precision=0.0,
            assertion_recall=0.0,
            assertion_precision=0.0,
            callable_recall=0.0,
            callable_precision=0.0,
            focal_recall=0.0,
            focal_precision=0.0,
        )

    @staticmethod
    def _zero_coverage_eval() -> NL2TestCoverageEval:
        return NL2TestCoverageEval(
            class_coverage=0.0,
            method_coverage=0.0,
            line_coverage=0.0,
            branch_coverage=0.0,
        )

    def _get_constructor_types(self, class_name: str, method_sig: str) -> Set[str]:
        types: Set[str] = set()
        try:
            method_details: Optional[JCallable] = self.analysis.get_method(
                qualified_class_name=class_name, qualified_method_name=method_sig
            )
        except Exception:
            method_details = None

        if not method_details:
            return types

        for cs in method_details.call_sites:
            if cs.is_constructor_call:
                ctor_type = cs.return_type
                if ctor_type:
                    types.add(ctor_type)

        return types

    def _get_focal_methods_for_test(
        self, qualified_class_name: str, method_signature: str
    ) -> Set[Tuple[str, str]]:
        """Collect focal (class, method) pairs for a given test method."""
        try:
            setup_methods = SetupAnalysisInfo(self.analysis).get_setup_methods(
                qualified_class_name
            )

            focal = FocalClassMethod(
                self.analysis, self.application_classes, self.test_utility_classes
            )
            focal_classes, _, _, _ = focal.extract_test_scope(
                qualified_class_name, method_signature, setup_methods
            )

            result: Set[Tuple[str, str]] = set()
            for fc in focal_classes:
                for method_name in fc.focal_method_names:
                    result.add((fc.focal_class, method_name))
            return result
        except Exception:
            return set()

    def grade_structural(
        self,
        pred_method_sig: str,
        pred_class_name: str,
        gt_method_sig: str,
        gt_class_name: str,
    ) -> Optional[NL2TestStructuralEval]:
        if not self.analysis.get_class(pred_class_name) or not self.analysis.get_method(
            pred_class_name, pred_method_sig
        ):
            return None

        gt_frameworks = self.common.get_testing_frameworks_for_class(gt_class_name)
        pred_frameworks = self.common.get_testing_frameworks_for_class(pred_class_name)

        gt_setup_methods: Dict[str, List[str]] = SetupAnalysisInfo(
            self.analysis
        ).get_setup_methods(gt_class_name)
        gt_analysis = TestMethodAnalysisInfo(
            self.analysis, "TestDataset", self.application_classes
        ).get_test_method_analysis_info(
            gt_frameworks, gt_class_name, gt_method_sig, gt_setup_methods
        )

        pred_setup_methods: Dict[str, List[str]] = SetupAnalysisInfo(
            self.analysis
        ).get_setup_methods(pred_class_name)
        pred_analysis = TestMethodAnalysisInfo(
            self.analysis, "TestDataset", self.application_classes
        ).get_test_method_analysis_info(
            pred_frameworks, pred_class_name, pred_method_sig, pred_setup_methods
        )

        # Object creation recall via constructor call-sites
        gt_ctor_types = self._get_constructor_types(gt_class_name, gt_method_sig)
        pred_ctor_types = self._get_constructor_types(pred_class_name, pred_method_sig)
        ctor_matches = gt_ctor_types & pred_ctor_types
        obj_creation_recall = (
            len(ctor_matches) / len(gt_ctor_types) if gt_ctor_types else 1.0
        )
        obj_creation_precision = (
            len(ctor_matches) / len(pred_ctor_types) if pred_ctor_types else 1.0
        )

        # Assertions and callables recall
        gt_assertion_types = self._get_assertion_types(
            gt_analysis.call_assertion_sequences or []
        )
        pred_assertion_types = self._get_assertion_types(
            pred_analysis.call_assertion_sequences or []
        )
        assertion_recall, assertion_precision = self._assertion_scores(
            gt_assertion_types, pred_assertion_types
        )
        callable_recall, callable_precision = self._callable_scores(
            gt_analysis.call_assertion_sequences or [],
            pred_analysis.call_assertion_sequences or [],
        )

        # Focal recall: number of GT focal methods also in pred focal set
        gt_focal = self._get_focal_methods_for_test(gt_class_name, gt_method_sig)
        pred_focal = self._get_focal_methods_for_test(pred_class_name, pred_method_sig)
        focal_intersection = gt_focal & pred_focal
        focal_recall = (len(focal_intersection) / len(gt_focal)) if gt_focal else 1.0
        focal_precision = (
            len(focal_intersection) / len(pred_focal) if pred_focal else 1.0
        )

        return NL2TestStructuralEval(
            obj_creation_recall=round(obj_creation_recall, 4),
            obj_creation_precision=round(obj_creation_precision, 4),
            assertion_recall=round(assertion_recall, 4),
            assertion_precision=round(assertion_precision, 4),
            callable_recall=round(callable_recall, 4),
            callable_precision=round(callable_precision, 4),
            focal_recall=round(focal_recall, 4),
            focal_precision=round(focal_precision, 4),
        )

    def grade_coverage(
        self,
        pred_method_sig: str,
        pred_class_name: str,
        gt_method_sig: str,
        gt_class_name: str,
    ) -> Optional[NL2TestCoverageEval]:
        """
        Compute coverage overlap between prediction and ground truth.
        Returns per-metric fractions in [0.0, 1.0] (non-penalizing 1.0 when GT is empty).
        """
        if not self.analysis.get_class(pred_class_name) or not self.analysis.get_method(
            pred_class_name, pred_method_sig
        ):
            return None

        # Run coverage for both tests
        tests_to_run = [
            (pred_class_name, pred_method_sig.split("(")[0]),
            (gt_class_name, gt_method_sig.split("(")[0]),
        ]
        module_root = self.common.resolve_module_root(
            gt_class_name
        ) or self.common.resolve_module_root(pred_class_name)
        try:
            coverage_runner = IndividualTestCoverage.from_common_analysis(
                self.common,
                project_root=self.project_root,
                qualified_class_names=[gt_class_name, pred_class_name],
                module_root=module_root,
            )
            all_coverage_details = coverage_runner.generate(tests_to_run=tests_to_run)
        except Exception as exc:
            if module_root is None:
                RichLog.warn(
                    "Coverage evaluation failed because module root could not be resolved "
                    f"for {gt_class_name} or {pred_class_name}: {exc}"
                )
            else:
                RichLog.warn(
                    f"Coverage evaluation failed for {pred_class_name}::{pred_method_sig}: {exc}"
                )
            return None

        gt_coverage_details: Dict[str, Dict[str, Any]] = {}
        pred_coverage_details: Dict[str, Dict[str, Any]] = {}

        if gt_class_name in all_coverage_details:
            for gt_method_coverage in all_coverage_details[gt_class_name]:
                if gt_method_coverage["test_name"] == gt_method_sig.split("(")[0]:
                    gt_coverage_details = gt_method_coverage["coverage_details"]

        if pred_class_name in all_coverage_details:
            for pred_method_coverage in all_coverage_details[pred_class_name]:
                if pred_method_coverage["test_name"] == pred_method_sig.split("(")[0]:
                    pred_coverage_details = pred_method_coverage["coverage_details"]

        def _as_set(x) -> Set[int]:
            if x is None:
                return set()
            if isinstance(x, set):
                return x
            if isinstance(x, (list, tuple)):
                return set(x)
            return {x}

        # Collect covered sets we'll compare
        def _collect(
            coverage_map: Dict[str, Dict[str, Any]],
        ) -> Tuple[
            Set[str],  # covered classes
            Set[Tuple[str, str]],  # covered methods keyed by (app_class, method_name)
            Set[Tuple[str, int]],  # covered lines keyed by (app_class, line)
            Set[Tuple[str, int]],  # covered branch lines keyed by (app_class, line)
        ]:
            covered_classes: Set[str] = set()
            covered_methods: Set[Tuple[str, str]] = set()
            covered_lines: Set[Tuple[str, int]] = set()
            covered_branch_lines: Set[Tuple[str, int]] = set()

            for app_cls, leaf in (coverage_map or {}).items():
                if not isinstance(leaf, dict):
                    continue

                # Methods
                for m in leaf.get("method_covered", []) or []:
                    covered_methods.add((app_cls, m))

                # Lines considered "covered": source lines + covered branches + partial branches
                c = _as_set(leaf.get("covered_lines"))
                bc = _as_set(leaf.get("branch_lines_covered"))
                bp = _as_set(leaf.get("branch_lines_partial"))

                # If any covered content exists in this class, mark class as covered
                if c or bc or bp:
                    covered_classes.add(app_cls)

                for ln in c | bc | bp:
                    covered_lines.add((app_cls, ln))

                # Branch coverage counts both fully and partially covered
                for ln in bc | bp:
                    covered_branch_lines.add((app_cls, ln))

            return covered_classes, covered_methods, covered_lines, covered_branch_lines

        gt_classes, gt_methods, gt_lines, gt_branch_lines = _collect(
            gt_coverage_details
        )
        pred_classes, pred_methods, pred_lines, pred_branch_lines = _collect(
            pred_coverage_details
        )

        # Intersections (hits)
        hit_classes = pred_classes & gt_classes
        hit_methods = pred_methods & gt_methods
        hit_lines = pred_lines & gt_lines
        hit_branch_lines = pred_branch_lines & gt_branch_lines

        def frac(n: int, d: int) -> float:
            # Non-penalizing when GT denominator is empty: return 1.0
            return (n / d) if d > 0 else 1.0

        return NL2TestCoverageEval(
            class_coverage=round(frac(len(hit_classes), len(gt_classes)), 4),
            method_coverage=round(frac(len(hit_methods), len(gt_methods)), 4),
            line_coverage=round(frac(len(hit_lines), len(gt_lines)), 4),
            branch_coverage=round(frac(len(hit_branch_lines), len(gt_branch_lines)), 4),
        )

    def grade(
        self,
        nl2_input: NL2TestInput,
        nl2_metadata: NL2TestMetadata,
        *,
        compiles: bool = True,
    ) -> Tuple[Optional[NL2TestStructuralEval], Optional[NL2TestCoverageEval]]:
        if not compiles:
            return self._zero_structural_eval(), self._zero_coverage_eval()

        pred_class_name = nl2_metadata.qualified_test_class_name
        # Prefer method signature from metadata when available. If that method
        # cannot be found in the analysis, fall back to the first discovered test
        # method, and if none exist, use the first declared method.
        pred_method_sig = (nl2_metadata.method_signature or "").strip()

        def _method_exists(sig: str) -> bool:
            if not sig:
                return False
            try:
                return bool(self.analysis.get_method(pred_class_name, sig))
            except Exception:
                return False

        if not _method_exists(pred_method_sig):
            test_methods = self.common.get_test_methods_in_class(pred_class_name)
            fallback_sig = ""
            if test_methods:
                for _, candidate_sig in test_methods:
                    if _method_exists(candidate_sig):
                        fallback_sig = candidate_sig
                        break
                else:
                    fallback_sig = test_methods[0][1]
            if not fallback_sig:
                pred_methods = list(
                    self.analysis.get_methods_in_class(pred_class_name) or []
                )
                for candidate in pred_methods:
                    candidate_sig = getattr(candidate, "signature", None) or str(
                        candidate
                    )
                    if _method_exists(candidate_sig):
                        fallback_sig = candidate_sig
                        break
                if not fallback_sig and pred_methods:
                    first = pred_methods[0]
                    fallback_sig = getattr(first, "signature", first)
            pred_method_sig = fallback_sig.strip()

        gt_class_name = nl2_input.qualified_class_name
        gt_method_sig = nl2_input.method_signature

        structural_eval = self.grade_structural(
            pred_method_sig=pred_method_sig,
            pred_class_name=pred_class_name,
            gt_method_sig=gt_method_sig,
            gt_class_name=gt_class_name,
        )
        coverage_eval = self.grade_coverage(
            pred_method_sig=pred_method_sig,
            pred_class_name=pred_class_name,
            gt_method_sig=gt_method_sig,
            gt_class_name=gt_class_name,
        )

        return structural_eval, coverage_eval
