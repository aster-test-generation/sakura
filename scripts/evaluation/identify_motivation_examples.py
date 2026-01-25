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

from nltest.utils.models.nl2test import (
    NL2TestCoverageEval,
    NL2TestEval,
    NL2TestStructuralEval,
    OutOfBoxAgentEval,
)

# Project root (3 levels up from scripts/evaluation/)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Input directories
NL2TEST_EVAL_DIR = ROOT_DIR / "outputs" / "raw_outputs" / "nl2test_gemini_pro_output" # for NL2Test
OTHER_AGENT_EVAL_DIR = ROOT_DIR / "outputs" / "raw_outputs" / "gemini_cli_pro_output"  # for other agent

# Output directory
OUTPUT_DIR = ROOT_DIR / "outputs" / "motivation"

# Evaluation file name
EVAL_FILE_NAME = "nl2test_evaluation_results.json"

# Default threshold for "significant" average gap (NL2Test - other)
DEFAULT_SIGNIFICANT_GAP_THRESHOLD = 0.2

# Type alias for evaluation entries (can be either NL2TestEval or OutOfBoxAgentEval)
EvalEntry = Union[NL2TestEval, OutOfBoxAgentEval]


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
                print(f"Warning: Validation error for entry {index} in {eval_file}: {e}")
                continue

    return entries_by_id


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
        nl2test_val > other_val and not math.isclose(nl2test_val, other_val, rel_tol=1e-9)
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


def find_motivation_examples(
    nl2test_dir: Path,
    other_agent_dir: Path,
    significant_gap_threshold: float = DEFAULT_SIGNIFICANT_GAP_THRESHOLD,
) -> Dict[str, Any]:
    """Find entries where NL2Test outperforms other agent.

    Identifies two categories:
    1. Compile Gap: NL2Test compiles, other agent doesn't (with failed_code_generation=False)
    2. Quality Gap: Both compile, NL2Test beats other agent on all 8 metrics
       - beats_other: All metrics >=, at least one strictly >
       - significant: Average gap across all metrics > significant_gap_threshold
       - perfect: All NL2Test metrics = 1.0 while beating other
       - perfect_significant: Perfect scores AND average gap > significant_gap_threshold

    Note: Entries without structural_eval or coverage_eval are skipped.

    Args:
        nl2test_dir: Directory with NL2Test evaluation results.
        other_agent_dir: Directory with other agent evaluation results.
        significant_gap_threshold: Minimum average gap to be considered "significant".

    Returns:
        Dictionary with summary and categorized entry IDs.
    """
    print(f"Loading NL2Test entries from: {nl2test_dir}")
    nl2test_entries = load_evaluation_results(nl2test_dir, use_out_of_box=False)
    print(f"Loaded {len(nl2test_entries)} NL2Test entries")

    print(f"Loading other agent entries from: {other_agent_dir}")
    other_entries = load_evaluation_results(other_agent_dir, use_out_of_box=True)
    print(f"Loaded {len(other_entries)} other agent entries")

    # Find matching IDs
    matching_ids = set(nl2test_entries.keys()) & set(other_entries.keys())
    print(f"Found {len(matching_ids)} matching entries")

    # Compile Gap buckets
    compile_gap_high_quality: List[int] = []
    compile_gap_perfect: List[int] = []

    # Quality Gap buckets
    quality_gap_beats_other: List[int] = []
    quality_gap_significant: List[int] = []
    quality_gap_perfect: List[int] = []
    quality_gap_perfect_significant: List[int] = []

    for entry_id in sorted(matching_ids):
        nl2test_entry = nl2test_entries[entry_id]
        other_entry = other_entries[entry_id]

        # Skip entries without complete eval data for NL2Test
        if not has_complete_eval_data(nl2test_entry):
            continue

        # other_entries is loaded with use_out_of_box=True, so all are OutOfBoxAgentEval
        assert isinstance(other_entry, OutOfBoxAgentEval)
        other_failed_code_gen = other_entry.failed_code_generation

        # Compile Gap Analysis: NL2Test compiles, other doesn't
        if nl2test_entry.compiles and not other_entry.compiles and not other_failed_code_gen:
            # Check for high quality (> 0.8)
            # We already checked that structural_eval and coverage_eval are not None
            assert nl2test_entry.structured_eval is not None
            assert nl2test_entry.coverage_eval is not None

            if (
                check_structural_recall_above(nl2test_entry.structured_eval, 0.8)
                and check_coverage_above(nl2test_entry.coverage_eval, 0.8)
            ):
                compile_gap_high_quality.append(entry_id)

                # Check for perfect (== 1.0)
                if (
                    check_structural_recall_equals(nl2test_entry.structured_eval, 1.0)
                    and check_coverage_equals(nl2test_entry.coverage_eval, 1.0)
                ):
                    compile_gap_perfect.append(entry_id)

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
                quality_gap_beats_other.append(entry_id)

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
                    quality_gap_significant.append(entry_id)

                # perfect: NL2Test achieves perfect scores (1.0) while beating other
                is_perfect = (
                    check_structural_recall_equals(nl2test_entry.structured_eval, 1.0)
                    and check_coverage_equals(nl2test_entry.coverage_eval, 1.0)
                )
                if is_perfect:
                    quality_gap_perfect.append(entry_id)

                    # perfect_significant: Perfect scores AND significant gap
                    if is_significant:
                        quality_gap_perfect_significant.append(entry_id)

    return {
        "summary": {
            "nl2test_eval_dir": str(nl2test_dir),
            "other_agent_eval_dir": str(other_agent_dir),
            "total_nl2test_entries": len(nl2test_entries),
            "total_other_agent_entries": len(other_entries),
            "matched_entries": len(matching_ids),
        },
        "compile_gap": {
            "description": "NL2Test compiles, other agent does not (with failed_code_generation=False)",
            "high_quality": {
                "description": "Structural recall > 0.8 and coverage > 0.8",
                "count": len(compile_gap_high_quality),
                "entry_ids": compile_gap_high_quality,
            },
            "perfect": {
                "description": "Structural recall = 1.0 and coverage = 1.0",
                "count": len(compile_gap_perfect),
                "entry_ids": compile_gap_perfect,
            },
        },
        "quality_gap": {
            "description": "Both compile, NL2Test beats other agent on all 8 metrics",
            "beats_other": {
                "description": "All NL2Test metrics >= other agent, at least one strictly greater",
                "count": len(quality_gap_beats_other),
                "entry_ids": quality_gap_beats_other,
            },
            "significant": {
                "description": f"Average gap across all 8 metrics > {significant_gap_threshold}",
                "count": len(quality_gap_significant),
                "entry_ids": quality_gap_significant,
            },
            "perfect": {
                "description": "All NL2Test metrics = 1.0 while beating other agent",
                "count": len(quality_gap_perfect),
                "entry_ids": quality_gap_perfect,
            },
            "perfect_significant": {
                "description": f"All metrics = 1.0 AND average gap > {significant_gap_threshold}",
                "count": len(quality_gap_perfect_significant),
                "entry_ids": quality_gap_perfect_significant,
            },
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
        "--significant-gap-threshold",
        type=float,
        default=DEFAULT_SIGNIFICANT_GAP_THRESHOLD,
        help=f"Minimum average gap to be considered 'significant' (default: {DEFAULT_SIGNIFICANT_GAP_THRESHOLD})",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    results = find_motivation_examples(
        args.nl2test_dir, args.other_agent_dir, args.significant_gap_threshold
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "motivation_examples.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to: {output_file}")
    print("\nSummary:")
    print(f"  Compile Gap - High Quality: {results['compile_gap']['high_quality']['count']}")
    print(f"  Compile Gap - Perfect: {results['compile_gap']['perfect']['count']}")
    print(f"  Quality Gap - Beats Other: {results['quality_gap']['beats_other']['count']}")
    print(f"  Quality Gap - Significant: {results['quality_gap']['significant']['count']}")
    print(f"  Quality Gap - Perfect: {results['quality_gap']['perfect']['count']}")
    print(f"  Quality Gap - Perfect Significant: {results['quality_gap']['perfect_significant']['count']}")


if __name__ == "__main__":
    main()
