from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import List, Tuple

DEFAULT_RESULTS_FILE = "tests/output/commons-cli/commons_cli_nl2test_evaluation_results.json"

# Tools to count (from composition agent)
LOCALIZATION_TOOLS = [
    "query_class_db",
]


def get_tool_counts_from_trajectories(
    trajectories: List[List[str]], tool_names: List[str]
) -> int:
    """Count occurrences of tools in trajectories."""
    count = 0
    for trajectory in trajectories:
        for tool_name in tool_names:
            count += trajectory.count(tool_name)
    return count


def pearson_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient between two lists."""
    if len(x) != len(y) or len(x) < 2:
        return float("nan")

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)

    denominator = math.sqrt(sum_sq_x * sum_sq_y)
    if denominator == 0:
        return float("nan")

    return numerator / denominator


def analyze_composition_tool_usage(results_file: Path) -> None:
    """Analyze composition agent tool usage from evaluation results."""
    with open(results_file, "r") as f:
        results = json.load(f)

    tool_counts: List[int] = []
    paired_data: List[Tuple[int, float]] = []

    for entry in results:
        tool_log = entry.get("tool_log")
        if tool_log is None:
            continue

        composition_log = tool_log.get("composition_tool_log")
        if composition_log is None:
            continue

        trajectories = composition_log.get("tool_trajectories", [])
        count = get_tool_counts_from_trajectories(trajectories, LOCALIZATION_TOOLS)
        tool_counts.append(count)

        # Extract localization_recall for correlation
        localization_eval = entry.get("localization_eval")
        if localization_eval is not None:
            localization_recall = localization_eval.get("localization_recall")
            if localization_recall is not None:
                paired_data.append((count, localization_recall))

    if not tool_counts:
        print("No composition tool logs found in the results file.")
        return

    # Calculate statistics
    avg = statistics.mean(tool_counts)
    median = statistics.median(tool_counts)
    sorted_counts = sorted(tool_counts)
    n = len(sorted_counts)

    q1 = statistics.median(sorted_counts[: n // 2])
    q3 = statistics.median(sorted_counts[(n + 1) // 2 :])
    min_val = min(tool_counts)
    max_val = max(tool_counts)

    # Print results
    print("=" * 60)
    print("  Localization Tool Use (Composition Agent)")
    print(f"  Tools: {', '.join(LOCALIZATION_TOOLS)}")
    print(f"  Source: {results_file}")
    print("=" * 60)
    print()
    print(f"  Total entries analyzed: {len(tool_counts)}")
    print()
    print("  Statistics:")
    print("  " + "-" * 40)
    print(f"  Average:      {avg:.2f}")
    print(f"  Median:       {median:.2f}")
    print()
    print("  Quartile Ranges:")
    print("  " + "-" * 40)
    print(f"  Min (Q0):     {min_val}")
    print(f"  Q1 (25%):     {q1:.2f}")
    print(f"  Q2 (50%):     {median:.2f}")
    print(f"  Q3 (75%):     {q3:.2f}")
    print(f"  Max (Q4):     {max_val}")
    print()

    # Calculate and display correlation with localization_recall
    if paired_data:
        tool_vals = [float(p[0]) for p in paired_data]
        recall_vals = [p[1] for p in paired_data]
        correlation = pearson_correlation(tool_vals, recall_vals)

        print("  Correlation Analysis:")
        print("  " + "-" * 40)
        print(f"  Paired samples:           {len(paired_data)}")
        print(f"  Pearson correlation:      {correlation:.4f}")
        print("  (localization_tool_use vs localization_recall)")
        print()

    print("=" * 60)


def main() -> None:
    root_dir = Path(__file__).parent.parent
    results_file = root_dir / DEFAULT_RESULTS_FILE

    if not results_file.exists():
        print(f"Error: Results file not found at {results_file}")
        return

    analyze_composition_tool_usage(results_file)


if __name__ == "__main__":
    main()
