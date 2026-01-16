"""Aggregate NL2Test evaluation results across project outputs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from pydantic import ValidationError

from nltest.nl2test.models.decomposition import LocalizationEval
from nltest.utils.models.nl2test import NL2TestEval, NL2TestInput

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT_DIR / "resources" / "agent_outputs" / "evaluation"
EVAL_FILE_NAME = "nl2test_evaluation_results.json"

ABSTRACTION_ORDER = ("high", "medium", "low")

STRUCTURAL_METRICS = (
    "obj_creation_recall",
    "obj_creation_precision",
    "assertion_recall",
    "assertion_precision",
    "callable_recall",
    "callable_precision",
    "focal_recall",
    "focal_precision",
)
COVERAGE_METRICS = (
    "class_coverage",
    "method_coverage",
    "line_coverage",
    "branch_coverage",
)
LOCALIZATION_METRICS = ("localization_recall",)
USAGE_METRICS = ("input_tokens", "output_tokens", "llm_calls")
ALL_METRICS = (
    *STRUCTURAL_METRICS,
    *COVERAGE_METRICS,
    *LOCALIZATION_METRICS,
    *USAGE_METRICS,
)

MetricValues = Dict[str, List[float]]


@dataclass(frozen=True)
class DistributionSummary:
    mean: float
    p25: float
    p50: float
    p75: float
    p90: float
    max: float
    count: int


MetricDistributions = Dict[str, DistributionSummary]
ProblemEntry = Tuple[int, str, str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate NL2Test evaluation results."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory containing project outputs.",
    )
    parser.add_argument(
        "--eval-file-name",
        type=str,
        default=EVAL_FILE_NAME,
        help="Evaluation file name inside each project directory.",
    )
    return parser.parse_args()


def normalize_output_dir(output_dir: Path) -> Path:
    if output_dir.is_absolute():
        return output_dir
    return (ROOT_DIR / output_dir).resolve()


def sort_abstraction_levels(levels: Iterable[str]) -> List[str]:
    return sorted(
        levels,
        key=lambda level: (
            ABSTRACTION_ORDER.index(level)
            if level in ABSTRACTION_ORDER
            else len(ABSTRACTION_ORDER)
        ),
    )


def normalize_abstraction_level(level: Any) -> str:
    if level is None:
        return "unknown"
    if isinstance(level, str):
        cleaned = level.strip()
        return cleaned if cleaned else "unknown"
    if hasattr(level, "value"):
        return str(level.value)
    return str(level)


def build_empty_metrics() -> MetricValues:
    return {metric: [] for metric in ALL_METRICS}


def build_problem_entry(nl2test_input: NL2TestInput) -> ProblemEntry:
    abstraction_level = normalize_abstraction_level(nl2test_input.abstraction_level)
    return (
        nl2test_input.id,
        nl2test_input.project_name,
        nl2test_input.qualified_class_name,
        nl2test_input.method_signature,
        abstraction_level,
    )


def load_eval_file(eval_file: Path) -> List[NL2TestEval]:
    try:
        with eval_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        print(f"Skipping {eval_file}: invalid JSON.")
        return []

    if not isinstance(data, list):
        print(f"Skipping {eval_file}: expected a list of entries.")
        return []

    results: List[NL2TestEval] = []
    for index, item in enumerate(data):
        try:
            results.append(NL2TestEval.model_validate(item))
        except ValidationError:
            print(f"Skipping entry {index} in {eval_file} due to validation error.")
    return results


def parse_localization_eval(value: Any) -> LocalizationEval | None:
    if value is None:
        return None
    if isinstance(value, LocalizationEval):
        return value
    try:
        return LocalizationEval.model_validate(value)
    except ValidationError:
        return None


def append_metrics(
    metrics_map: MetricValues, source: Any, metric_names: Iterable[str]
) -> None:
    for name in metric_names:
        metrics_map[name].append(float(getattr(source, name)))


def append_metrics_pair(
    overall_metrics: MetricValues,
    per_level_metrics: MetricValues,
    source: Any,
    metric_names: Iterable[str],
) -> None:
    append_metrics(overall_metrics, source, metric_names)
    append_metrics(per_level_metrics, source, metric_names)


def percentile(sorted_values: List[float], percentile_value: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (percentile_value / 100) * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return lower_value + weight * (upper_value - lower_value)


def summarize_distribution(values: List[float]) -> DistributionSummary:
    if not values:
        return DistributionSummary(
            mean=0.0, p25=0.0, p50=0.0, p75=0.0, p90=0.0, max=0.0, count=0
        )

    sorted_values = sorted(values)
    mean_value = sum(values) / len(values)
    return DistributionSummary(
        mean=mean_value,
        p25=percentile(sorted_values, 25),
        p50=percentile(sorted_values, 50),
        p75=percentile(sorted_values, 75),
        p90=percentile(sorted_values, 90),
        max=sorted_values[-1],
        count=len(values),
    )


def build_distributions(metrics: Mapping[str, List[float]]) -> MetricDistributions:
    return {name: summarize_distribution(values) for name, values in metrics.items()}


def print_main_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_category_header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def print_metric_summary(metric_name: str, summary: DistributionSummary) -> None:
    label = metric_name.replace("_", " ").title()
    print(
        f"  {label:<24} mean={summary.mean:>7.2f}  p25={summary.p25:>7.2f}  p50={summary.p50:>7.2f}  p75={summary.p75:>7.2f}  p90={summary.p90:>7.2f}  max={summary.max:>7.2f}  (n={summary.count})"
    )


def format_summary(summary: DistributionSummary) -> str:
    return (
        f"mean={summary.mean:>5.2f}  "
        f"p25={summary.p25:>5.2f}  "
        f"p50={summary.p50:>5.2f}  "
        f"p75={summary.p75:>5.2f}  "
        f"p90={summary.p90:>5.2f}  "
        f"(n={summary.count})"
    )


def print_metric_summary_by_level(
    metric_name: str,
    summaries_by_level: Mapping[str, DistributionSummary | None],
    levels: List[str],
) -> None:
    label = metric_name.replace("_", " ").title()
    print(f"  {label:<24}", end="")
    for level in levels:
        summary = summaries_by_level.get(level)
        if summary is None or summary.count == 0:
            print(f"  {level}: --/--    ", end="")
        else:
            print(f"  {level}: {summary.mean:>5.2f}/{summary.max:>5.2f}", end="")
    print()


def print_category(
    title: str, metrics: Iterable[str], distributions: MetricDistributions
) -> None:
    print_category_header(title)
    for metric_name in metrics:
        print_metric_summary(metric_name, distributions[metric_name])
    print()


def print_category_by_level(
    title: str,
    metrics: Iterable[str],
    distributions_by_level: Mapping[str, MetricDistributions],
    levels: List[str],
) -> None:
    print_category_header(title)
    for metric_name in metrics:
        summaries_by_level = {
            level: distributions_by_level.get(level, {}).get(metric_name)
            for level in levels
        }
        print_metric_summary_by_level(metric_name, summaries_by_level, levels)
    print()


def main() -> None:
    args = parse_args()
    output_dir = normalize_output_dir(args.output_dir)
    eval_file_name = args.eval_file_name

    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return

    project_dirs = [path for path in output_dir.iterdir() if path.is_dir()]
    if not project_dirs:
        print(f"No project directories found in {output_dir}")
        return

    eval_files: List[Path] = []
    projects_with_evals: List[str] = []
    for project_dir in sorted(project_dirs, key=lambda path: path.name):
        eval_file = project_dir / eval_file_name
        if eval_file.is_file():
            eval_files.append(eval_file)
            projects_with_evals.append(project_dir.name)

    if not eval_files:
        print(f"No {eval_file_name} files found under {output_dir}")
        return

    metrics = build_empty_metrics()
    metrics_by_level: Dict[str, MetricValues] = {}
    count_by_level: Dict[str, int] = {}
    problematic_entries: List[ProblemEntry] = []
    total_evals = 0
    compiles_count = 0
    valid_entries = 0
    missing_structured = 0
    missing_coverage = 0
    missing_localization = 0

    for eval_file in eval_files:
        eval_entries = load_eval_file(eval_file)
        for entry in eval_entries:
            total_evals += 1
            if entry.compiles:
                compiles_count += 1

            localization_eval = parse_localization_eval(entry.localization_eval)
            has_structured = entry.structured_eval is not None
            has_coverage = entry.coverage_eval is not None
            has_localization = localization_eval is not None

            if not has_structured:
                missing_structured += 1
            if not has_coverage:
                missing_coverage += 1
            if not has_localization:
                missing_localization += 1

            abstraction_level = normalize_abstraction_level(
                entry.nl2test_input.abstraction_level
            )
            if abstraction_level not in metrics_by_level:
                metrics_by_level[abstraction_level] = build_empty_metrics()
            count_by_level[abstraction_level] = (
                count_by_level.get(abstraction_level, 0) + 1
            )

            if has_structured and has_coverage and has_localization:
                valid_entries += 1

            if has_structured:
                append_metrics_pair(
                    metrics,
                    metrics_by_level[abstraction_level],
                    entry.structured_eval,
                    STRUCTURAL_METRICS,
                )
            if has_coverage:
                append_metrics_pair(
                    metrics,
                    metrics_by_level[abstraction_level],
                    entry.coverage_eval,
                    COVERAGE_METRICS,
                )
            if has_localization:
                append_metrics_pair(
                    metrics,
                    metrics_by_level[abstraction_level],
                    localization_eval,
                    LOCALIZATION_METRICS,
                )

            append_metrics_pair(
                metrics,
                metrics_by_level[abstraction_level],
                entry,
                USAGE_METRICS,
            )

            if not (has_structured and has_coverage and has_localization):
                problematic_entries.append(build_problem_entry(entry.nl2test_input))

    compile_rate = compiles_count / total_evals if total_evals else 0.0

    print_main_header("NL2Test Evaluation Summary")
    print(f"Output directory:    {output_dir}")
    print(f"Eval file:           {eval_file_name}")
    print(f"Projects scanned:    {len(project_dirs)}")
    print(f"Projects with evals: {len(eval_files)}")
    print(f"Total evaluations:   {total_evals}")
    print(f"Compiles:            {compiles_count} ({compile_rate:.1%})")
    print(f"Valid entries:       {valid_entries}")
    print(f"Problematic entries: {len(problematic_entries)}")
    print(f"Missing structured:  {missing_structured}")
    print(f"Missing coverage:    {missing_coverage}")
    print(f"Missing localization: {missing_localization}")

    print("\nProjects with evaluation results:")
    for project_name in projects_with_evals:
        print(f"  - {project_name}")

    if count_by_level:
        print("\nEntries by abstraction level:")
        for level in sort_abstraction_levels(count_by_level.keys()):
            print(f"  {level:<8} {count_by_level[level]}")

    if valid_entries == 0:
        print(
            "\nNo entries with complete structured, coverage, and localization evals."
        )

    overall_distributions = build_distributions(metrics)
    print_main_header("Holistic Distributions")
    print_category("Structured Metrics", STRUCTURAL_METRICS, overall_distributions)
    print_category("Coverage Metrics", COVERAGE_METRICS, overall_distributions)
    print_category("Localization Metrics", LOCALIZATION_METRICS, overall_distributions)
    print_category("Usage Metrics", USAGE_METRICS, overall_distributions)

    if not metrics_by_level:
        return

    levels = sort_abstraction_levels(metrics_by_level.keys())
    per_level_distributions = {
        level: build_distributions(level_metrics)
        for level, level_metrics in metrics_by_level.items()
    }

    print_main_header("Distributions By Abstraction Level")
    print_category_by_level(
        "Structured Metrics",
        STRUCTURAL_METRICS,
        per_level_distributions,
        levels,
    )
    print_category_by_level(
        "Coverage Metrics",
        COVERAGE_METRICS,
        per_level_distributions,
        levels,
    )
    print_category_by_level(
        "Localization Metrics",
        LOCALIZATION_METRICS,
        per_level_distributions,
        levels,
    )
    print_category_by_level(
        "Usage Metrics",
        USAGE_METRICS,
        per_level_distributions,
        levels,
    )


if __name__ == "__main__":
    main()
