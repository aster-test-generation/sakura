from __future__ import annotations

### DEPRECATED: Use new test_grader

from pathlib import Path
from typing import List, Set, Tuple, Dict, Any, Optional

from cldk.analysis.java import JavaAnalysis
from hamster.code_analysis.focal_class_method.focal_class_method import FocalClassMethod
from hamster.code_analysis.model.models import (
    CallAndAssertionSequenceDetails,
    AssertionType,
    AssertionDetails,
    CallableDetails,
)
from hamster.code_analysis.test_statistics import (
    TestMethodAnalysisInfo,
    SetupAnalysisInfo,
)


from nltest.utils.analysis import CommonAnalysis

from nltest.utils.coverage.individual_test_coverage import IndividualTestCoverage


class TestGraderOld:
    def __init__(
        self,
        analysis: JavaAnalysis,
        project_root: Path,
        project_erroneous_files: Optional[Set[str]] = None,
    ) -> None:
        self.analysis = analysis
        self.project_root = project_root
        # Track erroneous Java filenames (e.g., FooTest.java)
        self.erroneous_files = project_erroneous_files or set()
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
    def _exact_assertion_matching(
        gt_assertions: List[List[AssertionType]],
        pred_assertions: List[List[AssertionType]],
    ) -> float:
        pred_sets = [set(pred) for pred in pred_assertions]
        matches = sum(
            1 if set(gt) & ps else 0 for gt, ps in zip(gt_assertions, pred_sets)
        )
        return matches / len(gt_assertions) if gt_assertions else 0.0

    def _longest_callable_subsequence(
        self,
        gt_seqs: List[CallAndAssertionSequenceDetails],
        pred_seqs: List[CallAndAssertionSequenceDetails],
    ) -> float:
        gt_expanded = self._get_expanded_seqs(gt_seqs)
        pred_expanded = self._get_expanded_seqs(pred_seqs)

        def _key(seq: CallableDetails | AssertionDetails):
            if isinstance(seq, CallableDetails):
                return ("call", seq.method_name)
            if isinstance(seq, AssertionDetails):
                return ("assertion", seq.assertion_name)
            return None

        gt_keys = [k for k in (_key(s) for s in gt_expanded) if k is not None]
        pred_keys = [k for k in (_key(s) for s in pred_expanded) if k is not None]

        n, m = len(gt_keys), len(pred_keys)
        if n == 0 or m == 0:
            return 0.0

        # LCS DP implementation
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n):
            for j in range(m):
                if gt_keys[i] == pred_keys[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

        lcs_len = dp[n][m]
        return lcs_len / n

    def _callable_coverage(
        self,
        gt_seqs: List[CallAndAssertionSequenceDetails],
        pred_seqs: List[CallAndAssertionSequenceDetails],
    ) -> float:
        from collections import Counter

        gt_expanded = self._get_expanded_seqs(gt_seqs)
        pred_expanded = self._get_expanded_seqs(pred_seqs)

        gt_calls = Counter(
            seq.method_name for seq in gt_expanded if isinstance(seq, CallableDetails)
        )
        pred_calls = Counter(
            seq.method_name for seq in pred_expanded if isinstance(seq, CallableDetails)
        )

        gt_asserts = Counter(
            seq.assertion_name
            for seq in gt_expanded
            if isinstance(seq, AssertionDetails)
        )
        pred_asserts = Counter(
            seq.assertion_name
            for seq in pred_expanded
            if isinstance(seq, AssertionDetails)
        )

        matched_calls = (gt_calls & pred_calls).total()
        matched_asserts = (gt_asserts & pred_asserts).total()
        total_gt = sum(gt_calls.values()) + sum(gt_asserts.values())
        return (matched_calls + matched_asserts) / total_gt if total_gt else 0.0

    @staticmethod
    def _assertion_coverage(
        gt_assertions: List[List[AssertionType]],
        pred_assertions: List[List[AssertionType]],
    ) -> float:
        pred_groups = [set(group) for group in pred_assertions]
        visited = set()
        matched = 0
        for gt_group in gt_assertions:
            for i, pred_set in enumerate(pred_groups):
                if i in visited:
                    continue
                if any(a in pred_set for a in gt_group):
                    matched += 1
                    visited.add(i)
                    break
        return matched / len(gt_assertions) if gt_assertions else 0.0

    def grade_coverage_all(
        self,
        class_method_pairs: List[Tuple[Tuple[str, str], Tuple[str, str]]],
    ) -> list:
        """
        Calculate coverage for each generated test and the ground truth
        Args:
            class_method_pairs: List of tuples, where each tuple is a tuple of (pred_class_name, pred_method_name) and
            (gt_class_name, gt_method_name)

        Returns:
            list: Each element is a dict where each pred and ground truth pair overlap coverage is computed
        """
        coverage_details = []
        tests_to_run = []
        # Run the tests needed
        for class_method_pair in class_method_pairs:
            # Add pred tests
            tests_to_run.append((class_method_pair[0][0], class_method_pair[0][1]))
            # Add ground truth tests
            tests_to_run.append((class_method_pair[1][0], class_method_pair[1][1]))

        all_coverage_details = IndividualTestCoverage(
            project_root=self.project_root
        ).generate(tests_to_run=tests_to_run)
        for class_method_pair in class_method_pairs:
            # Get overall coverage
            coverage = self.grade_coverage(
                pred_method_name=class_method_pair[0][1],
                pred_class_name=class_method_pair[0][0],
                gt_class_name=class_method_pair[1][0],
                gt_method_name=class_method_pair[1][1],
                all_coverage_details=all_coverage_details,
            )
            # all_coverage_details=all_coverage_details)
            # Add to the list
            if coverage is not None:
                coverage_details.append(
                    {
                        "pred_class_name": class_method_pair[0][0],
                        "pred_method_signature": class_method_pair[0][1],
                        "gt_class_name": class_method_pair[1][0],
                        "gt_method_signature": class_method_pair[1][1],
                        "coverage_details": coverage,
                    }
                )

        return coverage_details

    @staticmethod
    def grade_coverage(
        pred_method_name: str,
        pred_class_name: str,
        gt_method_name: str,
        gt_class_name: str,
        all_coverage_details: dict,
    ) -> Dict[str, float]:
        """
        Compute the overall coverage between the prediction and ground truth
        Args:
            pred_method_name:
            pred_class_name:
            gt_method_name:
            gt_class_name:
            all_coverage_details:

        Returns:

        """
        gt_coverage_details = {}
        pred_coverage_details = {}
        if gt_class_name in all_coverage_details:
            for gt_method_coverage in all_coverage_details[gt_class_name]:
                if gt_method_coverage["test_name"] == gt_method_name:
                    gt_coverage_details = gt_method_coverage["coverage_details"]

        if pred_class_name in all_coverage_details:
            for pred_method_coverage in all_coverage_details[pred_class_name]:
                if pred_method_coverage["test_name"] == pred_method_name:
                    pred_coverage_details = pred_method_coverage["coverage_details"]

        def _as_set(x) -> Set[int]:
            if x is None:
                return set()
            if isinstance(x, set):
                return x
            if isinstance(x, (list, tuple)):
                return set(x)
            return {x}

        # Collect (from a single test's per-app-class map) the covered sets we’ll compare
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

        # Helper for safe percent
        def pct(n: int, d: int) -> float:
            return (n / d * 100.0) if d > 0 else 0.0

        return {
            "class_coverage_percent": round(pct(len(hit_classes), len(gt_classes)), 2),
            "method_coverage_percent": round(pct(len(hit_methods), len(gt_methods)), 2),
            "line_coverage_percent": round(pct(len(hit_lines), len(gt_lines)), 2),
            "branch_coverage_percent": round(
                pct(len(hit_branch_lines), len(gt_branch_lines)), 2
            ),
        }

    def grade_structural(
        self,
        pred_method_sig: str,
        pred_class_name: str,
        gt_method_sig: str,
        gt_class_name: str,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Grades structural similarity of a predicted test vs ground truth.
        Returns an average score and the per-metric breakdown.
        """
        _, application_classes = (
            self.common.get_test_methods_classes_and_application_classes()
        )

        gt_frameworks = self.common.get_testing_frameworks_for_class(gt_class_name)
        pred_frameworks = self.common.get_testing_frameworks_for_class(pred_class_name)

        gt_setup_methods = SetupAnalysisInfo(self.analysis).get_setup_methods(
            gt_class_name, testing_frameworks=gt_frameworks
        )
        gt_analysis = TestMethodAnalysisInfo(
            self.analysis, "TestDataset", application_classes
        ).get_test_method_analysis_info(
            gt_frameworks, gt_class_name, gt_method_sig, gt_setup_methods
        )

        pred_setup_methods = SetupAnalysisInfo(self.analysis).get_setup_methods(
            pred_class_name, testing_frameworks=pred_frameworks
        )
        pred_analysis = TestMethodAnalysisInfo(
            self.analysis, "TestDataset", application_classes
        ).get_test_method_analysis_info(
            pred_frameworks, pred_class_name, pred_method_sig, pred_setup_methods
        )

        gt_num_obj_created = gt_analysis.number_of_objects_created
        pred_num_obj_created = pred_analysis.number_of_objects_created

        gt_num_constructors = len(gt_analysis.constructor_call_details)
        pred_num_constructors = len(pred_analysis.constructor_call_details)

        gt_num_app_calls = len(gt_analysis.application_call_details)
        pred_num_app_calls = len(pred_analysis.application_call_details)

        gt_num_library_calls = len(gt_analysis.library_call_details)
        pred_num_library_calls = len(pred_analysis.library_call_details)

        gt_num_assertions = self._count_num_assertions(
            gt_analysis.call_assertion_sequences
        )
        pred_num_assertions = self._count_num_assertions(
            pred_analysis.call_assertion_sequences
        )

        gt_assertion_types = self._get_assertion_types(
            gt_analysis.call_assertion_sequences
        )
        pred_assertion_types = self._get_assertion_types(
            pred_analysis.call_assertion_sequences
        )

        exact_assertion_matching = self._exact_assertion_matching(
            gt_assertion_types, pred_assertion_types
        )
        assertion_coverage = self._assertion_coverage(
            gt_assertion_types, pred_assertion_types
        )

        longest_callable_subsequence = self._longest_callable_subsequence(
            gt_analysis.call_assertion_sequences, pred_analysis.call_assertion_sequences
        )
        callable_coverage = self._callable_coverage(
            gt_analysis.call_assertion_sequences, pred_analysis.call_assertion_sequences
        )

        pred_class_simple = pred_class_name.rsplit(".", 1)[-1] + ".java"
        compilation_score = 0.0 if pred_class_simple in self.erroneous_files else 1.0

        gt_counts = {
            "objects_created": gt_num_obj_created,
            "constructors": gt_num_constructors,
            "application_calls": gt_num_app_calls,
            "library_calls": gt_num_library_calls,
            "assertions": gt_num_assertions,
        }
        pred_counts = {
            "objects_created": pred_num_obj_created,
            "constructors": pred_num_constructors,
            "application_calls": pred_num_app_calls,
            "library_calls": pred_num_library_calls,
            "assertions": pred_num_assertions,
        }

        def coverage(gt: int, pred: int) -> float:
            return (pred / gt) if gt > 0 else 1.0

        metrics: Dict[str, float] = {
            k: coverage(gt_counts[k], pred_counts[k]) for k in gt_counts
        }
        metrics["order_aware_assertion_coverage"] = exact_assertion_matching
        metrics["ground_truth_assertion_coverage"] = assertion_coverage
        metrics["longest_callable_subsequence"] = longest_callable_subsequence
        metrics["ground_truth_callable_coverage"] = callable_coverage
        metrics["compilation_score"] = compilation_score

        # Add focal-method coverage as a component metric
        focal_score, _ = self.grade_focal_coverage_between_tests(
            pred_class_name, pred_method_sig, gt_class_name, gt_method_sig
        )
        metrics["focal_method_coverage"] = focal_score

        avg_score = sum(metrics.values()) / len(metrics)
        return avg_score, metrics

    def _get_focal_methods_for_test(
        self, qualified_class_name: str, method_signature: str
    ) -> Set[Tuple[str, str]]:
        try:
            testing_frameworks = self.common.get_testing_frameworks_for_class(
                qualified_class_name
            )
            setup_methods = self.common.get_setup_methods(qualified_class_name)
            setup_sigs = [m.signature for m in setup_methods]
            _, application_classes = (
                self.common.get_test_methods_classes_and_application_classes()
            )

            focal = FocalClassMethod(
                self.analysis, testing_frameworks, application_classes
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

    @staticmethod
    def _safe_ratio(numer: int, denom: int) -> float:
        # If there is no ground truth/denominator, consider coverage perfect
        return (numer / denom) if denom > 0 else 1.0

    def grade_focal_coverage_between_tests(
        self,
        test_a_class: str,
        test_a_method_sig: str,
        test_b_class: str,
        test_b_method_sig: str,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Grades focal class/method coverage between two test methods. Returns how well the focal methods from A cover
        focal methods of B.
        """
        a_focal = self._get_focal_methods_for_test(test_a_class, test_a_method_sig)
        b_focal = self._get_focal_methods_for_test(test_b_class, test_b_method_sig)

        intersection = a_focal & b_focal
        union = a_focal | b_focal

        coverage_b_by_a = self._safe_ratio(len(intersection), len(b_focal))
        coverage_a_by_b = self._safe_ratio(len(intersection), len(a_focal))
        jaccard = (len(intersection) / len(union)) if union else 1.0

        details: Dict[str, Any] = {
            "a_focal_count": len(a_focal),
            "b_focal_count": len(b_focal),
            "intersection_count": len(intersection),
            "union_count": len(union),
            "coverage_b_by_a": coverage_b_by_a,
            "coverage_a_by_b": coverage_a_by_b,
            "jaccard": jaccard,
            "a_focal_methods": [f"{c}.{m}" for c, m in sorted(a_focal)],
            "b_focal_methods": [f"{c}.{m}" for c, m in sorted(b_focal)],
            "intersection_methods": [f"{c}.{m}" for c, m in sorted(intersection)],
            "union_methods": [f"{c}.{m}" for c, m in sorted(union)],
        }

        # Primary score: A should maximally cover B
        return coverage_b_by_a, details
