from pathlib import Path
from typing import List, Counter, Set

from cldk.analysis.java import JavaAnalysis
from hamster.code_analysis.model.models import CallAndAssertionSequenceDetails, AssertionType, AssertionDetails, \
    CallableDetails
from hamster.code_analysis.test_statistics import TestClassAnalysisInfo, TestMethodAnalysisInfo, SetupAnalysisInfo

from nltest.utils.analysis import CommonAnalysis
from nltest.utils.pretty.prints import pretty_print


class TestGrader:
    def __init__(self, analysis: JavaAnalysis, project_root: Path, project_erroneous_files: List[str]):
        self.analysis = analysis
        self.project_root = project_root
        # List of erroneous Java source filenames (e.g., FooTest.java)
        self.erroneous_files = project_erroneous_files

    def _count_num_assertions(self, seqs: List[CallAndAssertionSequenceDetails]) -> int:
        num_assertions = 0
        for seq in seqs:
            num_assertions += len(seq.assertion_details)
        return num_assertions

    def _get_assertion_types(self, seqs: List[CallAndAssertionSequenceDetails]) -> List[List[AssertionType]]:
        assertion_types: List[List[AssertionType]] = []
        for seq in seqs:
            for assertion in seq.assertion_details:
                assertion_types.append(assertion.assertion_type)
        return assertion_types

    def _exact_assertion_matching(self,
                                  gt_assertions: List[List[AssertionType]],
                                  pred_assertions: List[List[AssertionType]]) -> float:
        pred_sets = [set(pred) for pred in pred_assertions]

        matching_score = sum(
            bool(set(gt) & ps) # Returns 1 if non-empty intersection between sets
            for gt, ps in zip(gt_assertions, pred_sets) # Zip trims to the lowest set
        )

        matching_score = matching_score / len(gt_assertions) if len(gt_assertions) > 0 else 0
        return matching_score

    def _get_expanded_seqs(self, seqs: List[CallAndAssertionSequenceDetails]) -> List[CallableDetails | AssertionDetails]:
        expanded_seqs: List[CallableDetails | AssertionDetails] = []
        for seq in seqs:
            expanded_seqs.extend(seq.call_sequence_details)
            expanded_seqs.extend(seq.assertion_details)
        return expanded_seqs

    def _longest_callable_subsequence(self,
                                      gt_seqs: List[CallAndAssertionSequenceDetails],
                                      pred_seqs: List[CallAndAssertionSequenceDetails], ) -> float:
        gt_expanded_seqs = self._get_expanded_seqs(gt_seqs)
        pred_expanded_seqs = self._get_expanded_seqs(pred_seqs)

        def _key(seq):
            if isinstance(seq, CallableDetails):
                return "call", seq.method_name
            elif isinstance(seq, AssertionDetails):
                return "assertion", seq.assertion_name
            else:
                return None

        gt_keys = [k for k in (_key(s) for s in gt_expanded_seqs) if k is not None]
        pred_keys = [k for k in (_key(s) for s in pred_expanded_seqs) if k is not None]

        n, m = len(gt_keys), len(pred_keys)
        # If there is no ground truth, return 0
        if n == 0:
            return 0.0

        dp = [[0] * (m + 1) for _ in range(n + 1)] # LCS DP implementation where dp[i][j] == LCS length of gt_keys[:i], pred_keys[:j]
        for i in range(n):
            for j in range(m):
                if gt_keys[i] == pred_keys[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

        lcs_len = dp[n][m]
        return lcs_len / n if m > 0 else 0.0

    def _callable_coverage(self,
                           gt_seqs: List[CallAndAssertionSequenceDetails],
                           pred_seqs: List[CallAndAssertionSequenceDetails],):
        gt_expanded_seqs = self._get_expanded_seqs(gt_seqs)
        pred_expanded_seqs = self._get_expanded_seqs(pred_seqs)

        gt_calls = Counter(
            seq.method_name
            for seq in gt_expanded_seqs
            if isinstance(seq, CallableDetails)
        )
        pred_calls = Counter(
            seq.method_name
            for seq in pred_expanded_seqs
            if isinstance(seq, CallableDetails)
        )

        gt_asserts = Counter(
            seq.assertion_name
            for seq in gt_expanded_seqs
            if isinstance(seq, AssertionDetails)
        )
        pred_asserts = Counter(
            seq.assertion_name
            for seq in pred_expanded_seqs
            if isinstance(seq, AssertionDetails)
        )

        # Computes sum of minima for each key
        matched_calls = (gt_calls & pred_calls).total()
        matched_asserts = (gt_asserts & pred_asserts).total()

        total_gt = sum(gt_calls.values()) + sum(gt_asserts.values())

        return (matched_calls + matched_asserts) / total_gt if total_gt else 0.0

    def _assertion_coverage(self,
                            gt_assertions: List[List[AssertionType]],
                            pred_assertions: List[List[AssertionType]]) -> float:
        pred_groups: List[Set[AssertionType]] = [set(group) for group in pred_assertions]
        visited_indices = set()
        matching = 0

        for gt_group in gt_assertions:
            for i, pred_set in enumerate(pred_groups):
                if i in visited_indices:
                    continue

                if any(a in pred_set for a in gt_group):
                    matching += 1
                    visited_indices.add(i)
                    break

        return matching / len(gt_assertions) if len(gt_assertions) > 0 else 0.0

    def grade(self, pred_method_sig: str, pred_class_name: str, gt_method_sig: str, gt_class_name: str):
        """Assigns a grade to the corresponding test trial."""

        gt_test_frameworks = CommonAnalysis(self.analysis).get_testing_frameworks_for_class(gt_class_name)
        pred_test_frameworks = CommonAnalysis(self.analysis).get_testing_frameworks_for_class(pred_class_name)

        gt_setup_methods = SetupAnalysisInfo(self.analysis).get_setup_methods(
            gt_class_name,
            testing_frameworks=gt_test_frameworks
        )
        gt_method_analysis = TestMethodAnalysisInfo(self.analysis, "TestDataset").get_test_method_analysis_info(
            gt_test_frameworks,
            gt_class_name,
            gt_method_sig,
            gt_setup_methods
        )
        pred_setup_methods = SetupAnalysisInfo(self.analysis).get_setup_methods(
            pred_class_name,
            testing_frameworks=pred_test_frameworks
        )
        pred_method_analysis = TestMethodAnalysisInfo(self.analysis, "TestDataset").get_test_method_analysis_info(
            pred_test_frameworks,
            pred_class_name,
            pred_method_sig,
            pred_setup_methods
        )

        gt_ncloc: int = gt_method_analysis.ncloc
        pred_ncloc: int = pred_method_analysis.ncloc

        gt_num_obj_created: int = gt_method_analysis.number_of_objects_created
        pred_num_obj_created: int = pred_method_analysis.number_of_objects_created

        gt_num_constructors: int = len(gt_method_analysis.constructor_call_details)
        pred_num_constructors: int = len(pred_method_analysis.constructor_call_details)

        gt_num_app_calls: int = len(gt_method_analysis.application_call_details)
        pred_num_app_calls: int = len(pred_method_analysis.application_call_details)

        gt_num_library_calls: int = len(gt_method_analysis.library_call_details)
        pred_num_library_calls: int = len(pred_method_analysis.library_call_details)

        gt_num_assertions: int = self._count_num_assertions(gt_method_analysis.call_assertion_sequences)
        pred_num_assertions: int = self._count_num_assertions(pred_method_analysis.call_assertion_sequences)

        gt_assertion_types: List[List[AssertionType]] = self._get_assertion_types(gt_method_analysis.call_assertion_sequences)
        pred_assertion_types: List[List[AssertionType]] = self._get_assertion_types(pred_method_analysis.call_assertion_sequences)
        exact_assertion_matching: float = self._exact_assertion_matching(gt_assertion_types, pred_assertion_types)
        assertion_coverage: float = self._assertion_coverage(gt_assertion_types, pred_assertion_types)

        longest_callable_subsequence: float = self._longest_callable_subsequence(gt_method_analysis.call_assertion_sequences, pred_method_analysis.call_assertion_sequences)
        callable_coverage: float = self._callable_coverage(gt_method_analysis.call_assertion_sequences, pred_method_analysis.call_assertion_sequences)

        pred_class = pred_class_name.rsplit(".", 1)[-1]
        pred_class_with_ext = pred_class + ".java"
        if pred_class_with_ext not in self.erroneous_files:
            compilation_score = 1.0
        else:
            compilation_score = 0.0

        gt_collection = {
            #"ncloc": gt_ncloc,
            "number of objects created": gt_num_obj_created,
            "number of constructors": gt_num_constructors,
            "number of application calls": gt_num_app_calls,
            "number of library calls": gt_num_library_calls,
            "number of assertions": gt_num_assertions
        }
        pred_collection = {
            #"ncloc": pred_ncloc,
            "number of objects created": pred_num_obj_created,
            "number of constructors": pred_num_constructors,
            "number of application calls": pred_num_app_calls,
            "number of library calls": pred_num_library_calls,
            "number of assertions": pred_num_assertions
        }

        def per_feature_similarity(gt, pred):
            max_num = max(gt, pred)
            if max_num == 0:
                return 1.0
            return min(gt, pred) / max_num

        def per_feature_coverage(gt, pred):
            return pred / gt if gt > pred else 1.0

        sims = {
            key: per_feature_coverage(gt_collection[key], pred_collection[key])
            for key in gt_collection
        }

        sims["order-aware assertion coverage"] = exact_assertion_matching
        sims["ground truth assertion coverage"] = assertion_coverage
        sims["longest sequence of callables of ground truth covered"] = longest_callable_subsequence
        sims["ground truth callable coverage"] = callable_coverage
        sims["compilation score"] = compilation_score
        pretty_print("scores", sims)

        score = sum(sims.values()) / len(sims)
        print("Averaged score:", score)
        return score
