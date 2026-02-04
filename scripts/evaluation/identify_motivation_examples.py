"""Identify motivation examples where NL2Test outperforms other agents.

This script finds cases where:
1. Compile Gap: NL2Test compiles but another agent fails to generate compilable code
2. Quality Gap: Both compile, but NL2Test has significantly higher quality metrics
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from pydantic import ValidationError

from sakura.dataset_creation.model import NL2TestDataset
from sakura.nl2test.models.decomposition import LocalizationEval
from sakura.utils.models.nl2test import (
    NL2TestCoverageEval,
    NL2TestEval,
    NL2TestStructuralEval,
    OutOfBoxAgentEval,
)

# Project root (3 levels up from scripts/evaluation/)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Input directories
NL2TEST_EVAL_DIR = (
    ROOT_DIR / "outputs" / "raw_outputs" / "nl2test_gemini_pro_output"
)  # for NL2Test
OTHER_AGENT_EVAL_DIR = (
    ROOT_DIR / "outputs" / "raw_outputs" / "gemini_cli_pro_output"
)  # for other agent

# Output directory
OUTPUT_DIR = ROOT_DIR / "outputs" / "motivation"

# Evaluation file name
EVAL_FILE_NAME = "nl2test_evaluation_results.json"

# Default threshold for "significant" average gap (NL2Test - other)
DEFAULT_SIGNIFICANT_GAP_THRESHOLD = 0.2

# Directory containing per-project bucketed datasets (nl2test.json files)
BUCKETED_DATASET_DIR = ROOT_DIR / "resources" / "filtered_bucketed_tests"

# Bucket names based on focal method count (last two original buckets merged)
BUCKET_ONE_FOCAL = "one_focal"
BUCKET_TWO_FOCAL = "two_focal"
BUCKET_THREE_TO_FIVE_FOCAL = "three_to_five_focal"
BUCKET_MORE_THAN_FIVE_FOCAL = "more_than_five_focal"
BUCKET_UNKNOWN = "unknown"

ALL_BUCKETS = [
    BUCKET_ONE_FOCAL,
    BUCKET_TWO_FOCAL,
    BUCKET_THREE_TO_FIVE_FOCAL,
    BUCKET_MORE_THAN_FIVE_FOCAL,
    BUCKET_UNKNOWN,
]

# Type alias for evaluation entries (can be either NL2TestEval or OutOfBoxAgentEval)
EvalEntry = Union[NL2TestEval, OutOfBoxAgentEval]

# Type alias for bucket lookup key: (project_name, qualified_class_name, method_signature)
BucketKey = Tuple[str, str, str]


def load_evaluation_results(
    eval_dir: Path, use_out_of_box: bool = False
) -> Dict[int, EvalEntry]:
    """Load all evaluation entries from directory, indexed by ID.

    Args:
        eval_dir: Directory containing project subdirectories with evaluation results.
        use_out_of_box: If True, parse as OutOfBoxAgentEval; otherwise NL2TestEval.

    Returns:
        Dictionary mapping entry IDs to their validated evaluation data.
    """
    entries_by_id: Dict[int, EvalEntry] = {}
    model_class = OutOfBoxAgentEval if use_out_of_box else NL2TestEval

    if not eval_dir.exists():
        print(f"Warning: Directory does not exist: {eval_dir}")
        return entries_by_id

    for project_dir in sorted(eval_dir.iterdir()):
        if not project_dir.is_dir():
            continue

        eval_file = project_dir / EVAL_FILE_NAME
        if not eval_file.exists():
            continue

        try:
            with eval_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {eval_file}")
            continue

        if not isinstance(data, list):
            print(f"Warning: Expected list in {eval_file}, got {type(data).__name__}")
            continue

        for index, entry_data in enumerate(data):
            if not isinstance(entry_data, dict):
                continue
            try:
                entry = model_class.model_validate(entry_data)
                entry_id = entry.nl2test_input.id
                if entry_id is not None and entry_id >= 0:
                    entries_by_id[entry_id] = entry
            except ValidationError as e:
                print(
                    f"Warning: Validation error for entry {index} in {eval_file}: {e}"
                )
                continue

    return entries_by_id


def load_bucket_mappings(bucketed_dir: Path) -> Dict[BucketKey, str]:
    """Load bucket mappings from per-project nl2test.json files.

    Iterates through project directories and loads NL2TestDataset files to build
    a lookup map from test identifiers to bucket names. The last two original
    buckets (5-10 and >10 focal methods) are merged into 'more_than_five_focal'.

    Args:
        bucketed_dir: Directory containing project subdirectories with nl2test.json.

    Returns:
        Dict mapping (project_name, qualified_class_name, method_signature) to bucket name.
    """
    bucket_mappings: Dict[BucketKey, str] = {}

    if not bucketed_dir.exists():
        print(f"Warning: Bucketed dataset directory does not exist: {bucketed_dir}")
        return bucket_mappings

    # Map from NL2TestDataset attribute name to our bucket constant
    attr_to_bucket = {
        "tests_with_one_focal_methods": BUCKET_ONE_FOCAL,
        "tests_with_two_focal_methods": BUCKET_TWO_FOCAL,
        "tests_with_more_than_two_to_five_focal_methods": BUCKET_THREE_TO_FIVE_FOCAL,
        "tests_with_more_than_five_to_ten_focal_methods": BUCKET_MORE_THAN_FIVE_FOCAL,
        "tests_with_more_than_ten_focal_methods": BUCKET_MORE_THAN_FIVE_FOCAL,
    }

    for project_dir in sorted(bucketed_dir.iterdir()):
        if not project_dir.is_dir():
            continue

        nl2test_file = project_dir / "nl2test.json"
        if not nl2test_file.exists():
            continue

        try:
            with nl2test_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            dataset = NL2TestDataset.model_validate(data)
        except Exception as e:
            print(f"Warning: Failed to load {nl2test_file}: {e}")
            continue

        project_name = dataset.dataset_name

        for attr_name, bucket_name in attr_to_bucket.items():
            tests = getattr(dataset, attr_name, [])
            for test in tests:
                key: BucketKey = (
                    project_name,
                    test.qualified_class_name,
                    test.method_signature,
                )
                bucket_mappings[key] = bucket_name

    print(f"Loaded bucket mappings for {len(bucket_mappings)} tests")
    return bucket_mappings


def get_bucket_for_entry(
    entry: EvalEntry,
    bucket_mappings: Dict[BucketKey, str],
) -> str:
    """Get the bucket name for an evaluation entry.

    Args:
        entry: Evaluation entry containing nl2test_input.
        bucket_mappings: Lookup map from test identifiers to bucket names.

    Returns:
        Bucket name, or BUCKET_UNKNOWN if not found in mappings.
    """
    key: BucketKey = (
        entry.nl2test_input.project_name,
        entry.nl2test_input.qualified_class_name,
        entry.nl2test_input.method_signature,
    )
    return bucket_mappings.get(key, BUCKET_UNKNOWN)


def create_empty_bucket_dict() -> Dict[str, List[int]]:
    """Create an empty dict with all bucket names initialized to empty lists."""
    return {bucket: [] for bucket in ALL_BUCKETS}


def check_structural_recall_above(
    structural_eval: NL2TestStructuralEval, threshold: float
) -> bool:
    """Check if all structural recall metrics are above threshold.

    Args:
        structural_eval: Structural evaluation metrics.
        threshold: Minimum threshold value (exclusive for > comparison).

    Returns:
        True if all recall metrics are above threshold, False otherwise.
    """
    return (
        structural_eval.obj_creation_recall > threshold
        and structural_eval.assertion_recall > threshold
        and structural_eval.callable_recall > threshold
        and structural_eval.focal_recall > threshold
    )


def check_structural_recall_below(
    structural_eval: NL2TestStructuralEval, threshold: float
) -> bool:
    """Check if all structural recall metrics are below threshold.

    Args:
        structural_eval: Structural evaluation metrics.
        threshold: Maximum threshold value (exclusive for < comparison).

    Returns:
        True if all recall metrics are below threshold, False otherwise.
    """
    return (
        structural_eval.obj_creation_recall < threshold
        and structural_eval.assertion_recall < threshold
        and structural_eval.callable_recall < threshold
        and structural_eval.focal_recall < threshold
    )


def check_structural_recall_equals(
    structural_eval: NL2TestStructuralEval, target: float
) -> bool:
    """Check if all structural recall metrics equal target value.

    Uses math.isclose() for floating-point comparison to avoid precision issues.

    Args:
        structural_eval: Structural evaluation metrics.
        target: Target value to match.

    Returns:
        True if all recall metrics equal target, False otherwise.
    """
    return (
        math.isclose(structural_eval.obj_creation_recall, target, rel_tol=1e-9)
        and math.isclose(structural_eval.assertion_recall, target, rel_tol=1e-9)
        and math.isclose(structural_eval.callable_recall, target, rel_tol=1e-9)
        and math.isclose(structural_eval.focal_recall, target, rel_tol=1e-9)
    )


def check_coverage_above(coverage_eval: NL2TestCoverageEval, threshold: float) -> bool:
    """Check if all coverage metrics are above threshold.

    Args:
        coverage_eval: Coverage evaluation metrics.
        threshold: Minimum threshold value (exclusive for > comparison).

    Returns:
        True if all coverage metrics are above threshold, False otherwise.
    """
    return (
        coverage_eval.class_coverage > threshold
        and coverage_eval.method_coverage > threshold
        and coverage_eval.line_coverage > threshold
        and coverage_eval.branch_coverage > threshold
    )


def check_coverage_below(coverage_eval: NL2TestCoverageEval, threshold: float) -> bool:
    """Check if all coverage metrics are below threshold.

    Args:
        coverage_eval: Coverage evaluation metrics.
        threshold: Maximum threshold value (exclusive for < comparison).

    Returns:
        True if all coverage metrics are below threshold, False otherwise.
    """
    return (
        coverage_eval.class_coverage < threshold
        and coverage_eval.method_coverage < threshold
        and coverage_eval.line_coverage < threshold
        and coverage_eval.branch_coverage < threshold
    )


def check_coverage_equals(coverage_eval: NL2TestCoverageEval, target: float) -> bool:
    """Check if all coverage metrics equal target value.

    Uses math.isclose() for floating-point comparison to avoid precision issues.

    Args:
        coverage_eval: Coverage evaluation metrics.
        target: Target value to match.

    Returns:
        True if all coverage metrics equal target, False otherwise.
    """
    return (
        math.isclose(coverage_eval.class_coverage, target, rel_tol=1e-9)
        and math.isclose(coverage_eval.method_coverage, target, rel_tol=1e-9)
        and math.isclose(coverage_eval.line_coverage, target, rel_tol=1e-9)
        and math.isclose(coverage_eval.branch_coverage, target, rel_tol=1e-9)
    )


def has_complete_eval_data(entry: EvalEntry) -> bool:
    """Check if entry has both structural_eval and coverage_eval."""
    return entry.structured_eval is not None and entry.coverage_eval is not None


def has_high_localization(entry: NL2TestEval, threshold: float = 0.8) -> bool:
    """Check if entry has localization_eval with localization_recall >= threshold.

    Args:
        entry: NL2TestEval entry (not OutOfBoxAgentEval, which has no localization).
        threshold: Minimum localization_recall value (default 0.8).

    Returns:
        True if localization_eval exists and localization_recall >= threshold.
    """
    if entry.localization_eval is None:
        return False

    # Handle both dict (from JSON) and LocalizationEval model
    if isinstance(entry.localization_eval, dict):
        recall = entry.localization_eval.get("localization_recall")
        if recall is None:
            return False
        return recall >= threshold
    elif isinstance(entry.localization_eval, LocalizationEval):
        return entry.localization_eval.localization_recall >= threshold

    return False


def _get_metric_pairs(
    nl2test_structural: NL2TestStructuralEval,
    nl2test_coverage: NL2TestCoverageEval,
    other_structural: NL2TestStructuralEval,
    other_coverage: NL2TestCoverageEval,
) -> List[Tuple[float, float]]:
    """Extract all 8 metric pairs for comparison between NL2Test and other agent.

    Args:
        nl2test_structural: NL2Test structural evaluation metrics.
        nl2test_coverage: NL2Test coverage evaluation metrics.
        other_structural: Other agent structural evaluation metrics.
        other_coverage: Other agent coverage evaluation metrics.

    Returns:
        List of (nl2test_value, other_value) tuples for all 8 metrics.
    """
    return [
        # Structural recall metrics
        (nl2test_structural.obj_creation_recall, other_structural.obj_creation_recall),
        (nl2test_structural.assertion_recall, other_structural.assertion_recall),
        (nl2test_structural.callable_recall, other_structural.callable_recall),
        (nl2test_structural.focal_recall, other_structural.focal_recall),
        # Coverage metrics
        (nl2test_coverage.class_coverage, other_coverage.class_coverage),
        (nl2test_coverage.method_coverage, other_coverage.method_coverage),
        (nl2test_coverage.line_coverage, other_coverage.line_coverage),
        (nl2test_coverage.branch_coverage, other_coverage.branch_coverage),
    ]


def nl2test_beats_other(
    nl2test_structural: NL2TestStructuralEval,
    nl2test_coverage: NL2TestCoverageEval,
    other_structural: NL2TestStructuralEval,
    other_coverage: NL2TestCoverageEval,
    min_gap: float = 0.0,
) -> bool:
    """Check if NL2Test beats other agent on all metrics with optional minimum gap.

    When min_gap == 0: Returns True if all NL2Test metrics >= other, with at least
    one strictly greater.
    When min_gap > 0: Returns True if all NL2Test metrics >= other + min_gap
    (the "at least one strictly greater" requirement is automatically satisfied).

    Uses math.isclose() for floating-point comparisons to avoid precision issues.

    Args:
        nl2test_structural: NL2Test structural evaluation metrics.
        nl2test_coverage: NL2Test coverage evaluation metrics.
        other_structural: Other agent structural evaluation metrics.
        other_coverage: Other agent coverage evaluation metrics.
        min_gap: Minimum gap required (NL2Test must be >= other + min_gap).

    Returns:
        True if NL2Test beats other on all metrics (with gap), False otherwise.
    """
    metric_pairs = _get_metric_pairs(
        nl2test_structural, nl2test_coverage, other_structural, other_coverage
    )

    # Check all NL2Test metrics >= other + min_gap (with float tolerance)
    target_vals = [other_val + min_gap for _, other_val in metric_pairs]
    all_at_least_gap = all(
        nl2test_val >= target or math.isclose(nl2test_val, target, rel_tol=1e-9)
        for (nl2test_val, _), target in zip(metric_pairs, target_vals)
    )

    if not all_at_least_gap:
        return False

    # When min_gap > 0, the gap requirement already ensures strict improvement
    if min_gap > 0:
        return True

    # When min_gap == 0, require at least one metric to be strictly greater
    # (greater than other AND not approximately equal)
    return any(
        nl2test_val > other_val
        and not math.isclose(nl2test_val, other_val, rel_tol=1e-9)
        for nl2test_val, other_val in metric_pairs
    )


def compute_average_gap(
    nl2test_structural: NL2TestStructuralEval,
    nl2test_coverage: NL2TestCoverageEval,
    other_structural: NL2TestStructuralEval,
    other_coverage: NL2TestCoverageEval,
) -> float:
    """Compute the average gap between NL2Test and other agent across all 8 metrics.

    Args:
        nl2test_structural: NL2Test structural evaluation metrics.
        nl2test_coverage: NL2Test coverage evaluation metrics.
        other_structural: Other agent structural evaluation metrics.
        other_coverage: Other agent coverage evaluation metrics.

    Returns:
        Average gap (NL2Test - other) across all 8 metrics.
    """
    metric_pairs = _get_metric_pairs(
        nl2test_structural, nl2test_coverage, other_structural, other_coverage
    )
    gaps = [nl2test_val - other_val for nl2test_val, other_val in metric_pairs]
    return sum(gaps) / len(gaps)


def _format_bucket_results(
    bucket_dict: Dict[str, List[int]], description: str
) -> Dict[str, Any]:
    """Format a bucket dictionary into the output JSON structure.

    Args:
        bucket_dict: Dict mapping bucket names to lists of entry IDs.
        description: Description of this category.

    Returns:
        Formatted dict with description, total_count, and by_bucket breakdown.
    """
    total_count = sum(len(ids) for ids in bucket_dict.values())
    by_bucket = {
        bucket: {"count": len(ids), "entry_ids": ids}
        for bucket, ids in bucket_dict.items()
    }
    return {
        "description": description,
        "total_count": total_count,
        "by_bucket": by_bucket,
    }


def find_motivation_examples(
    nl2test_dir: Path,
    other_agent_dir: Path,
    bucketed_dataset_dir: Path,
    significant_gap_threshold: float = DEFAULT_SIGNIFICANT_GAP_THRESHOLD,
) -> Dict[str, Any]:
    """Find entries where NL2Test outperforms other agent, organized by focal method bucket.

    Identifies two categories:
    1. Compile Gap: NL2Test compiles, other agent doesn't (with failed_code_generation=False)
    2. Quality Gap: Both compile, NL2Test beats other agent on all 8 metrics
       - beats_other: All metrics >=, at least one strictly >
       - significant: Average gap across all metrics > significant_gap_threshold
       - perfect: All NL2Test metrics = 1.0 while beating other
       - perfect_significant: Perfect scores AND average gap > significant_gap_threshold

    Results are organized by focal method bucket (one_focal, two_focal,
    three_to_five_focal, more_than_five_focal, unknown).

    Note: Entries without structural_eval or coverage_eval are skipped.

    Args:
        nl2test_dir: Directory with NL2Test evaluation results.
        other_agent_dir: Directory with other agent evaluation results.
        bucketed_dataset_dir: Directory with per-project nl2test.json bucket files.
        significant_gap_threshold: Minimum average gap to be considered "significant".

    Returns:
        Dictionary with summary and categorized entry IDs organized by bucket.
    """
    print(f"Loading NL2Test entries from: {nl2test_dir}")
    nl2test_entries = load_evaluation_results(nl2test_dir, use_out_of_box=False)
    print(f"Loaded {len(nl2test_entries)} NL2Test entries")

    print(f"Loading other agent entries from: {other_agent_dir}")
    other_entries = load_evaluation_results(other_agent_dir, use_out_of_box=True)
    print(f"Loaded {len(other_entries)} other agent entries")

    print(f"Loading bucket mappings from: {bucketed_dataset_dir}")
    bucket_mappings = load_bucket_mappings(bucketed_dataset_dir)

    # Find matching IDs
    matching_ids = set(nl2test_entries.keys()) & set(other_entries.keys())
    print(f"Found {len(matching_ids)} matching entries")

    # Compile Gap buckets (organized by focal method bucket)
    compile_gap_high_quality: Dict[str, List[int]] = create_empty_bucket_dict()
    compile_gap_perfect: Dict[str, List[int]] = create_empty_bucket_dict()
    compile_gap_perfect_high_localization: Dict[str, List[int]] = (
        create_empty_bucket_dict()
    )

    # Quality Gap buckets (organized by focal method bucket)
    quality_gap_beats_other: Dict[str, List[int]] = create_empty_bucket_dict()
    quality_gap_significant: Dict[str, List[int]] = create_empty_bucket_dict()
    quality_gap_perfect: Dict[str, List[int]] = create_empty_bucket_dict()
    quality_gap_perfect_significant: Dict[str, List[int]] = create_empty_bucket_dict()
    quality_gap_perfect_high_localization: Dict[str, List[int]] = (
        create_empty_bucket_dict()
    )
    quality_gap_perfect_significant_high_localization: Dict[str, List[int]] = (
        create_empty_bucket_dict()
    )

    for entry_id in sorted(matching_ids):
        nl2test_entry = nl2test_entries[entry_id]
        other_entry = other_entries[entry_id]

        # Skip entries without complete eval data for NL2Test
        if not has_complete_eval_data(nl2test_entry):
            continue

        # Determine the focal method bucket for this entry
        bucket = get_bucket_for_entry(nl2test_entry, bucket_mappings)

        # other_entries is loaded with use_out_of_box=True, so all are OutOfBoxAgentEval
        assert isinstance(other_entry, OutOfBoxAgentEval)
        other_failed_code_gen = other_entry.failed_code_generation

        # Compile Gap Analysis: NL2Test compiles, other doesn't
        if (
            nl2test_entry.compiles
            and not other_entry.compiles
            and not other_failed_code_gen
        ):
            # Check for high quality (> 0.8)
            # We already checked that structural_eval and coverage_eval are not None
            assert nl2test_entry.structured_eval is not None
            assert nl2test_entry.coverage_eval is not None

            if check_structural_recall_above(
                nl2test_entry.structured_eval, 0.8
            ) and check_coverage_above(nl2test_entry.coverage_eval, 0.8):
                compile_gap_high_quality[bucket].append(entry_id)

                # Check for perfect (== 1.0)
                if check_structural_recall_equals(
                    nl2test_entry.structured_eval, 1.0
                ) and check_coverage_equals(nl2test_entry.coverage_eval, 1.0):
                    compile_gap_perfect[bucket].append(entry_id)

                    # Check for perfect with perfect localization
                    if has_high_localization(nl2test_entry):
                        compile_gap_perfect_high_localization[bucket].append(entry_id)

        # Quality Gap Analysis: Both compile, but quality difference
        elif nl2test_entry.compiles and other_entry.compiles:
            # Skip if other entry doesn't have complete eval data
            if not has_complete_eval_data(other_entry):
                continue

            # We already checked that both have complete eval data
            assert nl2test_entry.structured_eval is not None
            assert nl2test_entry.coverage_eval is not None
            assert other_entry.structured_eval is not None
            assert other_entry.coverage_eval is not None

            # high_quality: NL2Test beats other agent on all 8 metrics (at least one strictly >)
            if nl2test_beats_other(
                nl2test_entry.structured_eval,
                nl2test_entry.coverage_eval,
                other_entry.structured_eval,
                other_entry.coverage_eval,
                min_gap=0.0,
            ):
                quality_gap_beats_other[bucket].append(entry_id)

                # Compute average gap for significant checks
                avg_gap = compute_average_gap(
                    nl2test_entry.structured_eval,
                    nl2test_entry.coverage_eval,
                    other_entry.structured_eval,
                    other_entry.coverage_eval,
                )

                # significant: Average gap across all metrics > threshold
                is_significant = avg_gap > significant_gap_threshold
                if is_significant:
                    quality_gap_significant[bucket].append(entry_id)

                # perfect: NL2Test achieves perfect scores (1.0) while beating other
                is_perfect = check_structural_recall_equals(
                    nl2test_entry.structured_eval, 1.0
                ) and check_coverage_equals(nl2test_entry.coverage_eval, 1.0)
                if is_perfect:
                    quality_gap_perfect[bucket].append(entry_id)

                    # perfect_significant: Perfect scores AND significant gap
                    if is_significant:
                        quality_gap_perfect_significant[bucket].append(entry_id)

                    # perfect_high_localization: Perfect scores AND perfect localization
                    if has_high_localization(nl2test_entry):
                        quality_gap_perfect_high_localization[bucket].append(entry_id)

                        # perfect_significant_high_localization: All three conditions
                        if is_significant:
                            quality_gap_perfect_significant_high_localization[
                                bucket
                            ].append(entry_id)

    return {
        "summary": {
            "nl2test_eval_dir": str(nl2test_dir),
            "other_agent_eval_dir": str(other_agent_dir),
            "bucketed_dataset_dir": str(bucketed_dataset_dir),
            "total_nl2test_entries": len(nl2test_entries),
            "total_other_agent_entries": len(other_entries),
            "matched_entries": len(matching_ids),
            "bucket_names": ALL_BUCKETS,
        },
        "compile_gap": {
            "description": "NL2Test compiles, other agent does not (with failed_code_generation=False)",
            "high_quality": _format_bucket_results(
                compile_gap_high_quality,
                "Structural recall > 0.8 and coverage > 0.8",
            ),
            "perfect": _format_bucket_results(
                compile_gap_perfect,
                "Structural recall = 1.0 and coverage = 1.0",
            ),
            "perfect_high_localization": _format_bucket_results(
                compile_gap_perfect_high_localization,
                "Structural recall = 1.0, coverage = 1.0, and localization_recall >= 0.8",
            ),
        },
        "quality_gap": {
            "description": "Both compile, NL2Test beats other agent on all 8 metrics",
            "beats_other": _format_bucket_results(
                quality_gap_beats_other,
                "All NL2Test metrics >= other agent, at least one strictly greater",
            ),
            "significant": _format_bucket_results(
                quality_gap_significant,
                f"Average gap across all 8 metrics > {significant_gap_threshold}",
            ),
            "perfect": _format_bucket_results(
                quality_gap_perfect,
                "All NL2Test metrics = 1.0 while beating other agent",
            ),
            "perfect_significant": _format_bucket_results(
                quality_gap_perfect_significant,
                f"All metrics = 1.0 AND average gap > {significant_gap_threshold}",
            ),
            "perfect_high_localization": _format_bucket_results(
                quality_gap_perfect_high_localization,
                "All metrics = 1.0 AND localization_recall >= 0.8",
            ),
            "perfect_significant_high_localization": _format_bucket_results(
                quality_gap_perfect_significant_high_localization,
                f"All metrics = 1.0, average gap > {significant_gap_threshold}, AND localization_recall >= 0.8",
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Identify motivation examples where NL2Test outperforms other agents."
    )
    parser.add_argument(
        "--nl2test-dir",
        type=Path,
        default=NL2TEST_EVAL_DIR,
        help="Directory containing NL2Test evaluation results",
    )
    parser.add_argument(
        "--other-agent-dir",
        type=Path,
        default=OTHER_AGENT_EVAL_DIR,
        help="Directory containing other agent evaluation results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write output JSON",
    )
    parser.add_argument(
        "--bucketed-dataset-dir",
        type=Path,
        default=BUCKETED_DATASET_DIR,
        help="Directory containing per-project nl2test.json bucket files",
    )
    parser.add_argument(
        "--significant-gap-threshold",
        type=float,
        default=DEFAULT_SIGNIFICANT_GAP_THRESHOLD,
        help=f"Minimum average gap to be considered 'significant' (default: {DEFAULT_SIGNIFICANT_GAP_THRESHOLD})",
    )
    return parser.parse_args()


def _print_bucket_summary(category_name: str, bucket_data: Dict[str, Any]) -> None:
    """Print summary for a category with per-bucket breakdown."""
    print(f"  {category_name}: {bucket_data['total_count']} total")
    for bucket_name in ALL_BUCKETS:
        count = bucket_data["by_bucket"][bucket_name]["count"]
        if count > 0:
            print(f"    - {bucket_name}: {count}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    results = find_motivation_examples(
        args.nl2test_dir,
        args.other_agent_dir,
        args.bucketed_dataset_dir,
        args.significant_gap_threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "motivation_examples.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to: {output_file}")
    print("\nSummary (by focal method bucket):")
    print("\nCompile Gap:")
    _print_bucket_summary("High Quality", results["compile_gap"]["high_quality"])
    _print_bucket_summary("Perfect", results["compile_gap"]["perfect"])
    _print_bucket_summary(
        "Perfect + High Localization",
        results["compile_gap"]["perfect_high_localization"],
    )
    print("\nQuality Gap:")
    _print_bucket_summary("Beats Other", results["quality_gap"]["beats_other"])
    _print_bucket_summary("Significant", results["quality_gap"]["significant"])
    _print_bucket_summary("Perfect", results["quality_gap"]["perfect"])
    _print_bucket_summary(
        "Perfect Significant", results["quality_gap"]["perfect_significant"]
    )
    _print_bucket_summary(
        "Perfect + High Localization",
        results["quality_gap"]["perfect_high_localization"],
    )
    _print_bucket_summary(
        "Perfect Significant + High Localization",
        results["quality_gap"]["perfect_significant_high_localization"],
    )


if __name__ == "__main__":
    main()
