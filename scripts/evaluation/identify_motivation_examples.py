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
from typing import Any, Dict, List, Union

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


def find_motivation_examples(
    nl2test_dir: Path,
    other_agent_dir: Path,
) -> Dict[str, Any]:
    """Find entries where NL2Test outperforms other agent.

    Identifies two categories:
    1. Compile Gap: NL2Test compiles, other agent doesn't (with failed_code_generation=False)
    2. Quality Gap: Both compile, but NL2Test quality high and other agent quality low

    Note: Entries without structural_eval or coverage_eval are skipped.

    Args:
        nl2test_dir: Directory with NL2Test evaluation results.
        other_agent_dir: Directory with other agent evaluation results.

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
    quality_gap_high: List[int] = []
    quality_gap_perfect: List[int] = []

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

            # Check if other agent has low quality (< 0.4)
            other_low = (
                check_structural_recall_below(other_entry.structured_eval, 0.4)
                and check_coverage_below(other_entry.coverage_eval, 0.4)
            )

            if not other_low:
                continue

            # NL2Test high quality (> 0.8)
            nl2test_high = (
                check_structural_recall_above(nl2test_entry.structured_eval, 0.8)
                and check_coverage_above(nl2test_entry.coverage_eval, 0.8)
            )

            if nl2test_high:
                quality_gap_high.append(entry_id)

                # Check for perfect (== 1.0)
                nl2test_perfect = (
                    check_structural_recall_equals(nl2test_entry.structured_eval, 1.0)
                    and check_coverage_equals(nl2test_entry.coverage_eval, 1.0)
                )
                if nl2test_perfect:
                    quality_gap_perfect.append(entry_id)

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
            "description": "Both compile, but NL2Test quality high and other agent low (<0.4)",
            "high_quality": {
                "description": "NL2Test structural recall > 0.8 and coverage > 0.8",
                "count": len(quality_gap_high),
                "entry_ids": quality_gap_high,
            },
            "perfect": {
                "description": "NL2Test structural recall = 1.0 and coverage = 1.0",
                "count": len(quality_gap_perfect),
                "entry_ids": quality_gap_perfect,
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
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    results = find_motivation_examples(args.nl2test_dir, args.other_agent_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "motivation_examples.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to: {output_file}")
    print("\nSummary:")
    print(f"  Compile Gap - High Quality: {results['compile_gap']['high_quality']['count']}")
    print(f"  Compile Gap - Perfect: {results['compile_gap']['perfect']['count']}")
    print(f"  Quality Gap - High Quality: {results['quality_gap']['high_quality']['count']}")
    print(f"  Quality Gap - Perfect: {results['quality_gap']['perfect']['count']}")


if __name__ == "__main__":
    main()
