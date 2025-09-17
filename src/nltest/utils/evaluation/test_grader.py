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

from nltest.nl2test.models.nl2test import (
    NL2TestCoverageEval,
    NL2TestInput,
    NL2TestMetadata,
    NL2TestEval,
    NL2TestStructuralEval,
)
from nltest.utils.analysis import CommonAnalysis
from nltest.utils.coverage.individual_test_coverage import IndividualTestCoverage


class TestGrader:
    """
    Grader for NL2Test structural and coverage evaluation.
    """

    def __init__(
        self,
        analysis: JavaAnalysis,
        project_root: Path,
        project_erroneous_classes: Optional[List[str]] = None,
        application_classes: Optional[List[str]] = None,
    ) -> None:
        self.analysis = analysis
        self.project_root = project_root
        self.project_erroneous_classes: Set[str] = set(project_erroneous_classes or [])
        self.application_classes: List[str] = list(application_classes or [])
        self.common = CommonAnalysis(analysis)

    def set_project_erroneous_classes(self, classes: List[str]) -> None:
        self.project_erroneous_classes = set(classes)

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
    def _assertion_recall(
        gt_assertions: List[List[AssertionType]],
        pred_assertions: List[List[AssertionType]],
    ) -> float:
        """Recall over assertion groups: how many GT groups are matched by any pred group."""
        pred_groups = [set(group) for group in pred_assertions]
        visited: Set[int] = set()
        matched = 0
        for gt_group in gt_assertions:
            for i, pred_set in enumerate(pred_groups):
                if i in visited:
                    continue
                if any(a in pred_set for a in gt_group):
                    matched += 1
                    visited.add(i)
                    break
        return matched / len(gt_assertions) if gt_assertions else 1.0

    @staticmethod
    def _callable_recall(
        gt_seqs: List[CallAndAssertionSequenceDetails],
        pred_seqs: List[CallAndAssertionSequenceDetails],
    ) -> float:
        """Recall for callable and assertions occurrences combined."""
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

        matched_asserts = 0
        used_pred_indices: Set[int] = set()
        for gt_types in gt_assert_types:
            for idx, pred_types in enumerate(pred_assert_types):
                if idx in used_pred_indices:
                    continue
                if any(a in pred_types for a in gt_types):
                    matched += 1
                    used_pred_indices.add(idx)
                    break

        matched_calls = (gt_calls & pred_calls).total()
        total_gt = sum(gt_calls.values()) + len(gt_assert_types)
        return (matched_calls + matched_asserts) / total_gt if total_gt else 1.0

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
            testing_frameworks = self.common.get_testing_frameworks_for_class(
                qualified_class_name
            )
            setup_methods = self.common.get_setup_methods(qualified_class_name)
            setup_sigs = [m.signature for m in setup_methods]

            focal = FocalClassMethod(
                self.analysis, testing_frameworks, self.application_classes
            )
            focal_classes, _, _, _ = focal.identify_focal_class_and_ui_api_test(
                qualified_class_name, method_signature, setup_sigs
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
    ) -> NL2TestStructuralEval:
        if not self.analysis.get_class(pred_class_name) or not self.analysis.get_method(
            pred_class_name, pred_method_sig
        ):
            return None

        gt_frameworks = self.common.get_testing_frameworks_for_class(gt_class_name)
        pred_frameworks = self.common.get_testing_frameworks_for_class(pred_class_name)

        gt_setup_methods = SetupAnalysisInfo(self.analysis).get_setup_methods(
            gt_class_name, testing_frameworks=gt_frameworks
        )
        gt_analysis = TestMethodAnalysisInfo(
            self.analysis, "TestDataset", self.application_classes
        ).get_test_method_analysis_info(
            gt_frameworks, gt_class_name, gt_method_sig, gt_setup_methods
        )

        pred_setup_methods = SetupAnalysisInfo(self.analysis).get_setup_methods(
            pred_class_name, testing_frameworks=pred_frameworks
        )
        pred_analysis = TestMethodAnalysisInfo(
            self.analysis, "TestDataset", self.application_classes
        ).get_test_method_analysis_info(
            pred_frameworks, pred_class_name, pred_method_sig, pred_setup_methods
        )

        # Object creation recall via constructor call-sites
        gt_ctor_types = self._get_constructor_types(gt_class_name, gt_method_sig)
        pred_ctor_types = self._get_constructor_types(pred_class_name, pred_method_sig)
        obj_creation_recall = (
            len(gt_ctor_types & pred_ctor_types) / len(gt_ctor_types)
            if gt_ctor_types
            else 1.0
        )

        # Assertions and callables recall
        gt_assertion_types = self._get_assertion_types(
            gt_analysis.call_assertion_sequences
        )
        pred_assertion_types = self._get_assertion_types(
            pred_analysis.call_assertion_sequences
        )
        assertion_recall = self._assertion_recall(
            gt_assertion_types, pred_assertion_types
        )
        callable_recall = self._callable_recall(
            gt_analysis.call_assertion_sequences, pred_analysis.call_assertion_sequences
        )

        # Focal recall: number of GT focal methods also in pred focal set
        gt_focal = self._get_focal_methods_for_test(gt_class_name, gt_method_sig)
        pred_focal = self._get_focal_methods_for_test(pred_class_name, pred_method_sig)
        focal_intersection = gt_focal & pred_focal
        focal_recall = (len(focal_intersection) / len(gt_focal)) if gt_focal else 1.0

        return NL2TestStructuralEval(
            obj_creation_recall=obj_creation_recall,
            assertion_recall=assertion_recall,
            callable_recall=callable_recall,
            focal_recall=focal_recall,
        )

    def grade_coverage(
        self,
        pred_method_sig: str,
        pred_class_name: str,
        gt_method_sig: str,
        gt_class_name: str,
    ) -> NL2TestCoverageEval:
        """
        Compute coverage overlap between prediction and ground truth.
        Returns coverage as percentages of GT covered by prediction.
        """
        if not self.analysis.get_class(pred_class_name) or not self.analysis.get_method(
            pred_class_name, pred_method_sig
        ):
            return None

        # Run coverage for both tests
        tests_to_run = [
            (pred_class_name, pred_method_sig),
            (gt_class_name, gt_method_sig),
        ]
        all_coverage_details = IndividualTestCoverage(
            project_root=self.project_root,
        ).generate(tests_to_run=tests_to_run)

        gt_coverage_details: Dict[str, Dict[str, Any]] = {}
        pred_coverage_details: Dict[str, Dict[str, Any]] = {}

        if gt_class_name in all_coverage_details:
            for gt_method_coverage in all_coverage_details[gt_class_name]:
                if gt_method_coverage["test_name"] == gt_method_sig:
                    gt_coverage_details = gt_method_coverage["coverage_details"]

        if pred_class_name in all_coverage_details:
            for pred_method_coverage in all_coverage_details[pred_class_name]:
                if pred_method_coverage["test_name"] == pred_method_sig:
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

        def pct(n: int, d: int) -> float:
            return (n / d * 100.0) if d > 0 else 0.0

        return NL2TestCoverageEval(
            class_coverage=round(pct(len(hit_classes), len(gt_classes)), 2),
            method_coverage=round(pct(len(hit_methods), len(gt_methods)), 2),
            line_coverage=round(pct(len(hit_lines), len(gt_lines)), 2),
            branch_coverage=round(pct(len(hit_branch_lines), len(gt_branch_lines)), 2),
        )

    def grade(
        self, nl2_input: NL2TestInput, nl2_metadata: NL2TestMetadata
    ) -> NL2TestEval:
        compiles = (
            nl2_metadata.qualified_test_class_name not in self.project_erroneous_classes
        )

        pred_class_name = nl2_metadata.qualified_test_class_name
        pred_methods = list(self.analysis.get_methods_in_class(pred_class_name) or [])
        if not pred_methods:
            pred_method_sig = ""
        else:
            first = pred_methods[0]
            pred_method_sig = getattr(first, "signature", first)

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

        return NL2TestEval(
            compiles=compiles,
            nl2test_input=nl2_input,
            nl2test_metadata=nl2_metadata,
            structured_eval=structural_eval,
            coverage_eval=coverage_eval,
        )
