"""
Script for basic terminal output of NL2Test evaluation results.
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_RESULTS_FILE = "tests/output/commons-cli/commons_cli_nl2test_evaluation_results.json"
# Set to True to show per-abstraction level statistics (grouped by abstraction_level)
# Set to False to show holistic statistics (all data combined)
SHOW_BY_ABSTRACTION = True
# ============================================================================

ABSTRACTION_ORDER = ["high", "medium", "low"]


def sort_abstraction_levels(levels: List[str]) -> List[str]:
    """Sort abstraction levels in the defined order (high, medium, low)."""
    return sorted(
        levels,
        key=lambda x: (
            ABSTRACTION_ORDER.index(x)
            if x in ABSTRACTION_ORDER
            else len(ABSTRACTION_ORDER)
        ),
    )


def load_evaluation_data(filepath: str) -> List[Dict]:
    """Load evaluation results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def count_compiles(data: List[Dict]) -> Tuple[int, int]:
    """Count the number of successful and failed compilations."""
    compiles_true = sum(1 for obj in data if obj.get("compiles", False))
    compiles_false = sum(1 for obj in data if not obj.get("compiles", False))
    return compiles_true, compiles_false


def group_by_abstraction(data: List[Dict]) -> Dict[str, List[Dict]]:
    """Group data by abstraction level."""
    grouped = {}
    for obj in data:
        abstraction = obj.get("nl2test_input", {}).get("abstraction_level", "unknown")
        if abstraction not in grouped:
            grouped[abstraction] = []
        grouped[abstraction].append(obj)
    return grouped


def collect_metrics(data: List[Dict]) -> Dict[str, List[float]]:
    """Extract all metrics from evaluation data."""
    metrics = {
        # Structured eval metrics
        "obj_creation_recall": [],
        "obj_creation_precision": [],
        "assertion_recall": [],
        "assertion_precision": [],
        "callable_recall": [],
        "callable_precision": [],
        "focal_recall": [],
        "focal_precision": [],
        # Coverage eval metrics
        "class_coverage": [],
        "method_coverage": [],
        "line_coverage": [],
        "branch_coverage": [],
        # Localization eval metric
        "localization_recall": [],
    }

    for obj in data:
        # Extract structured_eval metrics
        if "structured_eval" in obj:
            for key in [
                "obj_creation_recall",
                "obj_creation_precision",
                "assertion_recall",
                "assertion_precision",
                "callable_recall",
                "callable_precision",
                "focal_recall",
                "focal_precision",
            ]:
                if key in obj["structured_eval"]:
                    metrics[key].append(obj["structured_eval"][key])

        # Extract coverage_eval metrics
        if "coverage_eval" in obj:
            for key in [
                "class_coverage",
                "method_coverage",
                "line_coverage",
                "branch_coverage",
            ]:
                if key in obj["coverage_eval"]:
                    metrics[key].append(obj["coverage_eval"][key])

        # Extract localization_recall
        if (
            "localization_eval" in obj
            and "localization_recall" in obj["localization_eval"]
        ):
            metrics["localization_recall"].append(
                obj["localization_eval"]["localization_recall"]
            )

    return metrics


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate mean, median, and quartiles for a list of values."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "q1": 0.0, "q3": 0.0, "count": 0}

    sorted_values = sorted(values)
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "q1": (
            statistics.quantiles(sorted_values, n=4)[0]
            if len(values) >= 2
            else sorted_values[0]
        ),
        "q3": (
            statistics.quantiles(sorted_values, n=4)[2]
            if len(values) >= 2
            else sorted_values[-1]
        ),
        "count": len(values),
    }


def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_metric_stats(metric_name: str, stats: Dict[str, float]):
    """Print statistics for a single metric in a formatted way."""
    print(f"\n  {metric_name.replace('_', ' ').title()}")
    print(f"  {'-' * len(metric_name)}")
    print(f"    Mean:        {stats['mean']:.4f}")
    print(f"    Median:      {stats['median']:.4f}")
    print(f"    Q1 (25th):   {stats['q1']:.4f}")
    print(f"    Q3 (75th):   {stats['q3']:.4f}")


def print_metric_stats_by_abstraction(
    metric_name: str, abstraction_stats: Dict[str, Dict[str, float]]
):
    """Print statistics for a single metric across all abstraction levels."""
    print(f"\n  {metric_name.replace('_', ' ').title()}")
    print(f"  {'-' * len(metric_name)}")
    for abstraction_level in sort_abstraction_levels(list(abstraction_stats.keys())):
        stats = abstraction_stats[abstraction_level]
        print(f"    {abstraction_level}:")
        print(f"      Mean:        {stats['mean']:.4f}")
        print(f"      Median:      {stats['median']:.4f}")
        print(f"      Q1 (25th):   {stats['q1']:.4f}")
        print(f"      Q3 (75th):   {stats['q3']:.4f}")


def display_metrics_for_dataset(data: List[Dict], title_suffix: str = ""):
    """Display all metrics for a given dataset."""
    # Collect metrics
    metrics = collect_metrics(data)

    # Structured Evaluation Metrics
    print_section_header(f"STRUCTURED EVALUATION METRICS{title_suffix}")
    for metric in [
        "obj_creation_recall",
        "obj_creation_precision",
        "assertion_recall",
        "assertion_precision",
        "callable_recall",
        "callable_precision",
        "focal_recall",
        "focal_precision",
    ]:
        stats = calculate_statistics(metrics[metric])
        print_metric_stats(metric, stats)

    # Coverage Evaluation Metrics
    print_section_header(f"COVERAGE EVALUATION METRICS{title_suffix}")
    for metric in [
        "class_coverage",
        "method_coverage",
        "line_coverage",
        "branch_coverage",
    ]:
        stats = calculate_statistics(metrics[metric])
        print_metric_stats(metric, stats)

    # Localization Evaluation Metrics
    print_section_header(f"LOCALIZATION EVALUATION METRICS{title_suffix}")
    stats = calculate_statistics(metrics["localization_recall"])
    print_metric_stats("localization_recall", stats)


def display_metrics_by_abstraction(grouped_data: Dict[str, List[Dict]]):
    """Display all metrics grouped by metric name, showing all abstraction levels together."""
    # Collect metrics for each abstraction level
    all_metrics_by_abstraction = {}
    for abstraction_level, data in grouped_data.items():
        all_metrics_by_abstraction[abstraction_level] = collect_metrics(data)

    # Get all abstraction levels in the correct order
    abstraction_levels = sort_abstraction_levels(list(grouped_data.keys()))

    # Structured Evaluation Metrics
    print_section_header("STRUCTURED EVALUATION METRICS")
    for metric in [
        "obj_creation_recall",
        "obj_creation_precision",
        "assertion_recall",
        "assertion_precision",
        "callable_recall",
        "callable_precision",
        "focal_recall",
        "focal_precision",
    ]:
        abstraction_stats = {}
        for level in abstraction_levels:
            stats = calculate_statistics(all_metrics_by_abstraction[level][metric])
            abstraction_stats[level] = stats
        print_metric_stats_by_abstraction(metric, abstraction_stats)

    # Coverage Evaluation Metrics
    print_section_header("COVERAGE EVALUATION METRICS")
    for metric in [
        "class_coverage",
        "method_coverage",
        "line_coverage",
        "branch_coverage",
    ]:
        abstraction_stats = {}
        for level in abstraction_levels:
            stats = calculate_statistics(all_metrics_by_abstraction[level][metric])
            abstraction_stats[level] = stats
        print_metric_stats_by_abstraction(metric, abstraction_stats)

    # Localization Evaluation Metrics
    print_section_header("LOCALIZATION EVALUATION METRICS")
    abstraction_stats = {}
    for level in abstraction_levels:
        stats = calculate_statistics(
            all_metrics_by_abstraction[level]["localization_recall"]
        )
        abstraction_stats[level] = stats
    print_metric_stats_by_abstraction("localization_recall", abstraction_stats)


def main():
    """Main function to analyze and display evaluation results."""
    root_dir = Path(__file__).parent.parent
    filepath = root_dir / DEFAULT_RESULTS_FILE

    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "NL2Test Evaluation Results Analysis" + " " * 23 + "║")
    print("╚" + "═" * 78 + "╝")

    # Load data
    print("\n📊 Loading evaluation data...")
    data = load_evaluation_data(filepath)
    print(f"✓ Loaded {len(data)} evaluation results")

    # Count compilation results
    compiles_true, compiles_false = count_compiles(data)
    print(f"\n📝 Compilation Results:")
    print(f"   ✓ Compiles: {compiles_true} ({compiles_true/len(data)*100:.1f}%)")
    print(f"   ✗ Fails:    {compiles_false} ({compiles_false/len(data)*100:.1f}%)")

    if SHOW_BY_ABSTRACTION:
        # Group by abstraction level and display stats for each
        print("\n📊 Mode: PER-ABSTRACTION ANALYSIS")
        grouped = group_by_abstraction(data)

        # Display compilation stats for each abstraction level
        print("\n" + "=" * 80)
        print("  COMPILATION RESULTS BY ABSTRACTION LEVEL")
        print("=" * 80)
        for abstraction_level in sort_abstraction_levels(list(grouped.keys())):
            count = len(grouped[abstraction_level])
            compiles_true_abs, compiles_false_abs = count_compiles(
                grouped[abstraction_level]
            )
            print(f"\n  {abstraction_level}:")
            print(f"    Total:    {count}")
            print(
                f"    Compiles: {compiles_true_abs} ({compiles_true_abs/count*100:.1f}%)"
            )
            print(
                f"    Fails:    {compiles_false_abs} ({compiles_false_abs/count*100:.1f}%)"
            )

        # Display metrics grouped by metric name
        display_metrics_by_abstraction(grouped)
    else:
        # Display holistic stats
        print("\n📊 Mode: HOLISTIC ANALYSIS")
        display_metrics_for_dataset(data)

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"\n  Total evaluations processed: {len(data)}")
    print(f"  Compilation success rate: {compiles_true/len(data)*100:.1f}%")
    if SHOW_BY_ABSTRACTION:
        grouped = group_by_abstraction(data)
        sorted_levels = sort_abstraction_levels(list(grouped.keys()))
        print(f"  Abstraction levels found: {', '.join(sorted_levels)}")
        for level in sorted_levels:
            print(f"    - {level}: {len(grouped[level])} evaluations")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
