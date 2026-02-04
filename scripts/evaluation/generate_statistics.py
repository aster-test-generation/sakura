"""Aggregate NL2Test evaluation results across project outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from pydantic import ValidationError

from sakura.dataset_creation.model import NL2TestDataset, Test
from sakura.nl2test.models.decomposition import LocalizationEval
from sakura.utils.models.nl2test import NL2TestInput, OutOfBoxAgentEval
from sakura.utils.statistics import (
    DistributionSummary,
    build_distributions,
    distributions_to_dict,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
EVAL_FILE_NAME = "nl2test_evaluation_results.json"
OUTPUT_DIR = ROOT_DIR / "outputs" / "evaluation_stats"
BUCKETED_FILTERED_DATASET_DIR = ROOT_DIR / "resources" / "filtered_bucketed_tests"
BUCKETED_DATASET_FILE = "nl2test.json"
RAW_OUTPUTS_DIR = ROOT_DIR / "outputs" / "raw_outputs"

EVAL_DIR_PRICING: Dict[Path, str] = {
    RAW_OUTPUTS_DIR / "gemini_cli_pro_output": "gemini-2.5-pro",
    RAW_OUTPUTS_DIR / "gemini_cli_flash_output": "gemini-2.5-flash",
    RAW_OUTPUTS_DIR / "nl2test_gemini_flash_output": "gemini-2.5-flash",
    RAW_OUTPUTS_DIR / "nl2test_gemini_pro_output": "gemini-2.5-pro",
    RAW_OUTPUTS_DIR / "nl2test_devstral_output": "devstral-small-latest",
    RAW_OUTPUTS_DIR / "nl2test_qwen3_output": "qwen/qwen3-coder",
}
EVAL_DIRS: List[Path] = list(EVAL_DIR_PRICING.keys())

ABSTRACTION_ORDER = ("high", "medium", "low")

# Ablation settings
IGNORE_IF_NOT_COMPILE = False  # Only include entries that compile in all EVAL_DIRS
ONLY_SHARED_ENTRIES = (
    True  # Only include entries present in all EVAL_DIRS (allows non-compiling)
)

# Pricing per million tokens (USD)
MODEL_PRICING = {
    "gemini-2.5-flash": {
        "input_per_million": 0.30,
        "output_per_million": 2.50,
    },
    "gemini-2.5-pro": {
        "input_per_million_under_200k": 1.25,
        "input_per_million_over_200k": 2.50,
        "output_per_million_under_200k": 10.00,
        "output_per_million_over_200k": 15.00,
        "context_threshold": 200_000,
    },
    "minimax/minimax-m2.1": {
        "input_per_million": 0.27,
        "output_per_million": 1.12,
    },
    "xiaomi/mimo-v2-flash": {
        "input_per_million": 0.10,
        "output_per_million": 0.30,
    },
    "qwen/qwen3-coder": {
        "input_per_million": 0.22,
        "output_per_million": 0.95,
    },
    "devstral-small-latest": {
        "input_per_million": 0.1,
        "output_per_million": 0.3,
    },
    "deepseek/deepseek-v3.2": {
        "input_per_million": 0.25,
        "output_per_million": 0.38,
    },
}
FOCAL_BUCKET_ORDER = (
    "one_focal",
    "two_focal",
    "three_to_five_focal",
    "more_than_five_focal",
)
FOCAL_BUCKET_NAMES = {
    "one_focal": "1 Focal Method",
    "two_focal": "2 Focal Methods",
    "three_to_five_focal": "3-5 Focal Methods",
    "more_than_five_focal": ">5 Focal Methods",
}
FOCAL_METHOD_COUNT_ORDER = (
    "1_focal_method",
    "2_focal_methods",
    "3_focal_methods",
    "4_focal_methods",
    "5_focal_methods",
    "6_focal_methods",
    "7_focal_methods",
    "8_focal_methods",
    "9_focal_methods",
    "10_plus_focal_methods",
)

STRUCTURAL_METRICS = (
    "obj_creation_recall",
    "obj_creation_precision",
    "obj_creation_f1",
    "assertion_recall",
    "assertion_precision",
    "assertion_f1",
    "callable_recall",
    "callable_precision",
    "callable_f1",
    "focal_recall",
    "focal_precision",
    "focal_f1",
)
F1_METRIC_PAIRS = (
    ("obj_creation_precision", "obj_creation_recall", "obj_creation_f1"),
    ("assertion_precision", "assertion_recall", "assertion_f1"),
    ("callable_precision", "callable_recall", "callable_f1"),
    ("focal_precision", "focal_recall", "focal_f1"),
)
COVERAGE_METRICS = (
    "class_coverage",
    "method_coverage",
    "line_coverage",
    "branch_coverage",
)
LOCALIZATION_METRICS = ("localization_recall",)
ENTRY_USAGE_METRICS = ("input_tokens", "output_tokens", "llm_calls")
USAGE_METRICS = (*ENTRY_USAGE_METRICS, "cost")
ALL_METRICS = (
    *STRUCTURAL_METRICS,
    *COVERAGE_METRICS,
    *LOCALIZATION_METRICS,
    *USAGE_METRICS,
)

MetricValues = Dict[str, List[float]]
MetricDistributions = Dict[str, DistributionSummary]
ProblemEntry = Tuple[int, str, str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate NL2Test evaluation results."
    )
    parser.add_argument(
        "--eval-file-name",
        type=str,
        default=EVAL_FILE_NAME,
        help="Evaluation file name inside each project directory.",
    )
    return parser.parse_args()


def calculate_entry_cost(
    input_tokens: float, output_tokens: float, pricing_model: str
) -> float:
    """Calculate cost for a single entry based on token counts and pricing model."""
    pricing = MODEL_PRICING[pricing_model]

    if "input_per_million" in pricing and "output_per_million" in pricing:
        input_cost = (input_tokens / 1_000_000) * pricing["input_per_million"]
        output_cost = (output_tokens / 1_000_000) * pricing["output_per_million"]
    else:
        threshold = pricing["context_threshold"]
        if input_tokens <= threshold:
            input_cost = (input_tokens / 1_000_000) * pricing[
                "input_per_million_under_200k"
            ]
            output_cost = (output_tokens / 1_000_000) * pricing[
                "output_per_million_under_200k"
            ]
        else:
            input_cost = (input_tokens / 1_000_000) * pricing[
                "input_per_million_over_200k"
            ]
            output_cost = (output_tokens / 1_000_000) * pricing[
                "output_per_million_over_200k"
            ]

    return input_cost + output_cost


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


def sort_focal_buckets(buckets: Iterable[str]) -> List[str]:
    return sorted(
        buckets,
        key=lambda bucket: (
            FOCAL_BUCKET_ORDER.index(bucket)
            if bucket in FOCAL_BUCKET_ORDER
            else len(FOCAL_BUCKET_ORDER)
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


def load_bucketed_dataset(project_name: str) -> NL2TestDataset | None:
    dataset_file = BUCKETED_FILTERED_DATASET_DIR / project_name / BUCKETED_DATASET_FILE
    if not dataset_file.exists():
        return None
    try:
        with dataset_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return NL2TestDataset.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        return None


BucketTestMap = Dict[str, Tuple[Test, str]]


def build_bucket_test_map(dataset: NL2TestDataset) -> BucketTestMap:
    """Build a map of (qualified_class_name, method_signature) -> (Test, bucket_name)."""
    bucket_mapping: BucketTestMap = {}
    buckets = [
        (dataset.tests_with_one_focal_methods, "one_focal"),
        (dataset.tests_with_two_focal_methods, "two_focal"),
        (dataset.tests_with_more_than_two_to_five_focal_methods, "three_to_five_focal"),
        (
            dataset.tests_with_more_than_five_to_ten_focal_methods,
            "more_than_five_focal",
        ),
        (dataset.tests_with_more_than_ten_focal_methods, "more_than_five_focal"),
    ]
    for tests, bucket_name in buckets:
        for test in tests:
            key = f"{test.qualified_class_name}::{test.method_signature}"
            bucket_mapping[key] = (test, bucket_name)
    return bucket_mapping


def get_focal_class_count(test: Test) -> int:
    if test.focal_details is None:
        return 0
    return len(test.focal_details)


def normalize_focal_class_count(count: int) -> str:
    if count == 0:
        return "0_focal_classes"
    if count == 1:
        return "1_focal_class"
    if count == 2:
        return "2_focal_classes"
    if count <= 5:
        return "3_to_5_focal_classes"
    return "more_than_5_focal_classes"


FOCAL_CLASS_COUNT_ORDER = (
    "0_focal_classes",
    "1_focal_class",
    "2_focal_classes",
    "3_to_5_focal_classes",
    "more_than_5_focal_classes",
)


def sort_focal_class_counts(counts: Iterable[str]) -> List[str]:
    return sorted(
        counts,
        key=lambda count: (
            FOCAL_CLASS_COUNT_ORDER.index(count)
            if count in FOCAL_CLASS_COUNT_ORDER
            else len(FOCAL_CLASS_COUNT_ORDER)
        ),
    )


def get_total_focal_method_count(test: Test) -> int:
    """Count total focal methods across all focal classes."""
    if test.focal_details is None:
        return 0
    return sum(len(fci.focal_method_names) for fci in test.focal_details)


def normalize_focal_method_count(count: int) -> str:
    """Normalize focal method count to bucket key."""
    if count <= 0:
        return "0_focal_methods"
    if count >= 10:
        return "10_plus_focal_methods"
    if count == 1:
        return "1_focal_method"
    return f"{count}_focal_methods"


def sort_focal_method_counts(counts: Iterable[str]) -> List[str]:
    return sorted(
        counts,
        key=lambda c: (
            FOCAL_METHOD_COUNT_ORDER.index(c)
            if c in FOCAL_METHOD_COUNT_ORDER
            else len(FOCAL_METHOD_COUNT_ORDER)
        ),
    )


def load_eval_file(eval_file: Path) -> List[OutOfBoxAgentEval]:
    try:
        with eval_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        print(f"Skipping {eval_file}: invalid JSON.")
        return []

    if not isinstance(data, list):
        print(f"Skipping {eval_file}: expected a list of entries.")
        return []

    results: List[OutOfBoxAgentEval] = []
    for index, item in enumerate(data):
        try:
            results.append(OutOfBoxAgentEval.model_validate(item))
        except ValidationError:
            print(f"Skipping entry {index} in {eval_file} due to validation error.")
    return results


def build_ignore_ids_from_dirs(
    dirs: List[Path], eval_file_name: str, require_compile: bool = True
) -> set[int]:
    """Build a set of IDs to ignore based on presence (and optionally compilation) across directories.

    Returns IDs that are not in the minimal spanning set across all directories.
    If require_compile is True, an ID is ignored if it fails to compile OR is missing in ANY directory.
    If require_compile is False, an ID is ignored only if it is missing in ANY directory.
    """
    ids_that_compile: Dict[Path, set[int]] = {}
    ids_present: Dict[Path, set[int]] = {}

    for eval_dir in dirs:
        if not eval_dir.exists():
            print(f"Warning: Directory not found: {eval_dir}")
            ids_that_compile[eval_dir] = set()
            ids_present[eval_dir] = set()
            continue

        project_dirs = [path for path in eval_dir.iterdir() if path.is_dir()]
        compiling_ids: set[int] = set()
        present_ids: set[int] = set()

        for project_dir in project_dirs:
            eval_file = project_dir / eval_file_name
            if not eval_file.is_file():
                continue
            entries = load_eval_file(eval_file)
            for entry in entries:
                entry_id = entry.nl2test_input.id
                present_ids.add(entry_id)
                if entry.compiles:
                    compiling_ids.add(entry_id)

        ids_that_compile[eval_dir] = compiling_ids
        ids_present[eval_dir] = present_ids

    # All IDs that exist in any directory
    all_ids = set.union(*ids_present.values()) if ids_present else set()

    # IDs that compile in ALL directories
    if ids_that_compile:
        ids_compiling_in_all = set.intersection(*ids_that_compile.values())
    else:
        ids_compiling_in_all = set()

    # IDs that are present in ALL directories (minimal spanning set)
    if ids_present:
        ids_present_in_all = set.intersection(*ids_present.values())
    else:
        ids_present_in_all = set()

    # Valid IDs: present in all, and optionally compile in all
    if require_compile:
        valid_ids = ids_compiling_in_all & ids_present_in_all
    else:
        valid_ids = ids_present_in_all

    # Ignore IDs: all IDs across directories minus valid ones
    ignore_ids = all_ids - valid_ids

    print("\nIgnore IDs calculation:")
    print(f"  Total unique IDs across directories: {len(all_ids)}")
    print(f"  Require compile: {require_compile}")
    for eval_dir in dirs:
        dir_name = eval_dir.name
        present_count = len(ids_present.get(eval_dir, set()))
        compile_count = len(ids_that_compile.get(eval_dir, set()))
        print(f"  {dir_name}: {present_count} present, {compile_count} compile")
    print(f"  IDs compiling in all: {len(ids_compiling_in_all)}")
    print(f"  IDs present in all (minimal spanning set): {len(ids_present_in_all)}")
    if require_compile:
        print(f"  Valid IDs (present AND compile in all): {len(valid_ids)}")
    else:
        print(f"  Valid IDs (present in all): {len(valid_ids)}")
    print(f"  Ignored IDs: {len(ignore_ids)}")

    return ignore_ids


def parse_localization_eval(value: Any) -> LocalizationEval | None:
    if value is None:
        return None
    if isinstance(value, LocalizationEval):
        return value
    try:
        return LocalizationEval.model_validate(value)
    except ValidationError:
        return None


def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def append_structural_metrics_with_f1(metrics_map: MetricValues, source: Any) -> None:
    """Append structural metrics including calculated F1 scores."""
    for precision_name, recall_name, f1_name in F1_METRIC_PAIRS:
        precision = float(getattr(source, precision_name))
        recall = float(getattr(source, recall_name))
        metrics_map[precision_name].append(precision)
        metrics_map[recall_name].append(recall)
        metrics_map[f1_name].append(calculate_f1(precision, recall))


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


def evaluate_directory(
    output_dir: Path,
    eval_file_name: str,
    pricing_model: str,
    ignore_ids: set[int],
) -> None:
    """Process one directory and write its JSON evaluation output."""
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
    metrics_by_focal_bucket: Dict[str, MetricValues] = {}
    metrics_by_focal_class_count: Dict[str, MetricValues] = {}
    metrics_by_focal_method_count: Dict[str, MetricValues] = {}
    count_by_level: Dict[str, int] = {}
    count_by_focal_bucket: Dict[str, int] = {}
    count_by_focal_class_count: Dict[str, int] = {}
    count_by_focal_method_count: Dict[str, int] = {}
    compiles_by_level: Dict[str, int] = {}
    compiles_by_focal_bucket: Dict[str, int] = {}
    compiles_by_focal_class_count: Dict[str, int] = {}
    compiles_by_focal_method_count: Dict[str, int] = {}
    problematic_entries: List[ProblemEntry] = []
    non_compiling_ids: List[int] = []
    skipped_ids: List[int] = []
    total_evals = 0
    compiles_count = 0
    valid_entries = 0
    missing_structured = 0
    missing_coverage = 0
    missing_localization = 0
    failed_code_generation_count = 0
    failed_test_file_generation_count = 0

    bucketed_datasets: Dict[str, NL2TestDataset | None] = {}
    bucket_test_maps: Dict[str, BucketTestMap] = {}

    for eval_file in eval_files:
        eval_entries = load_eval_file(eval_file)
        for entry in eval_entries:
            entry_id = entry.nl2test_input.id

            if entry_id in ignore_ids:
                skipped_ids.append(entry_id)
                continue

            total_evals += 1
            if entry.compiles:
                compiles_count += 1
            else:
                non_compiling_ids.append(entry_id)

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

            code_is_empty = (
                not entry.nl2test_metadata.code
                or not entry.nl2test_metadata.code.strip()
            )
            if entry.failed_code_generation or code_is_empty:
                failed_code_generation_count += 1
            if entry.failed_test_file_generation:
                failed_test_file_generation_count += 1

            abstraction_level = normalize_abstraction_level(
                entry.nl2test_input.abstraction_level
            )
            if abstraction_level not in metrics_by_level:
                metrics_by_level[abstraction_level] = build_empty_metrics()
            count_by_level[abstraction_level] = (
                count_by_level.get(abstraction_level, 0) + 1
            )
            if entry.compiles:
                compiles_by_level[abstraction_level] = (
                    compiles_by_level.get(abstraction_level, 0) + 1
                )

            project_name = entry.nl2test_input.project_name
            if project_name not in bucketed_datasets:
                bucketed_datasets[project_name] = load_bucketed_dataset(project_name)
                dataset = bucketed_datasets[project_name]
                if dataset is not None:
                    bucket_test_maps[project_name] = build_bucket_test_map(dataset)

            focal_bucket: str | None = None
            focal_class_count_key: str | None = None
            focal_method_count_key: str | None = None
            test_key = f"{entry.nl2test_input.qualified_class_name}::{entry.nl2test_input.method_signature}"
            if project_name in bucket_test_maps:
                bucket_map = bucket_test_maps[project_name]
                if test_key in bucket_map:
                    test_entry, focal_bucket = bucket_map[test_key]
                    focal_class_count_key = normalize_focal_class_count(
                        get_focal_class_count(test_entry)
                    )
                    focal_method_count_key = normalize_focal_method_count(
                        get_total_focal_method_count(test_entry)
                    )

            if focal_bucket is not None:
                if focal_bucket not in metrics_by_focal_bucket:
                    metrics_by_focal_bucket[focal_bucket] = build_empty_metrics()
                count_by_focal_bucket[focal_bucket] = (
                    count_by_focal_bucket.get(focal_bucket, 0) + 1
                )
                if entry.compiles:
                    compiles_by_focal_bucket[focal_bucket] = (
                        compiles_by_focal_bucket.get(focal_bucket, 0) + 1
                    )

            if focal_class_count_key is not None:
                if focal_class_count_key not in metrics_by_focal_class_count:
                    metrics_by_focal_class_count[focal_class_count_key] = (
                        build_empty_metrics()
                    )
                count_by_focal_class_count[focal_class_count_key] = (
                    count_by_focal_class_count.get(focal_class_count_key, 0) + 1
                )
                if entry.compiles:
                    compiles_by_focal_class_count[focal_class_count_key] = (
                        compiles_by_focal_class_count.get(focal_class_count_key, 0) + 1
                    )

            if focal_method_count_key is not None:
                if focal_method_count_key not in metrics_by_focal_method_count:
                    metrics_by_focal_method_count[focal_method_count_key] = (
                        build_empty_metrics()
                    )
                count_by_focal_method_count[focal_method_count_key] = (
                    count_by_focal_method_count.get(focal_method_count_key, 0) + 1
                )
                if entry.compiles:
                    compiles_by_focal_method_count[focal_method_count_key] = (
                        compiles_by_focal_method_count.get(focal_method_count_key, 0)
                        + 1
                    )

            if has_structured and has_coverage and has_localization:
                valid_entries += 1

            if has_structured:
                append_structural_metrics_with_f1(metrics, entry.structured_eval)
                append_structural_metrics_with_f1(
                    metrics_by_level[abstraction_level], entry.structured_eval
                )
                if focal_bucket is not None:
                    append_structural_metrics_with_f1(
                        metrics_by_focal_bucket[focal_bucket],
                        entry.structured_eval,
                    )
                if focal_class_count_key is not None:
                    append_structural_metrics_with_f1(
                        metrics_by_focal_class_count[focal_class_count_key],
                        entry.structured_eval,
                    )
                if focal_method_count_key is not None:
                    append_structural_metrics_with_f1(
                        metrics_by_focal_method_count[focal_method_count_key],
                        entry.structured_eval,
                    )
            if has_coverage:
                append_metrics_pair(
                    metrics,
                    metrics_by_level[abstraction_level],
                    entry.coverage_eval,
                    COVERAGE_METRICS,
                )
                if focal_bucket is not None:
                    append_metrics(
                        metrics_by_focal_bucket[focal_bucket],
                        entry.coverage_eval,
                        COVERAGE_METRICS,
                    )
                if focal_class_count_key is not None:
                    append_metrics(
                        metrics_by_focal_class_count[focal_class_count_key],
                        entry.coverage_eval,
                        COVERAGE_METRICS,
                    )
                if focal_method_count_key is not None:
                    append_metrics(
                        metrics_by_focal_method_count[focal_method_count_key],
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
                if focal_bucket is not None:
                    append_metrics(
                        metrics_by_focal_bucket[focal_bucket],
                        localization_eval,
                        LOCALIZATION_METRICS,
                    )
                if focal_class_count_key is not None:
                    append_metrics(
                        metrics_by_focal_class_count[focal_class_count_key],
                        localization_eval,
                        LOCALIZATION_METRICS,
                    )
                if focal_method_count_key is not None:
                    append_metrics(
                        metrics_by_focal_method_count[focal_method_count_key],
                        localization_eval,
                        LOCALIZATION_METRICS,
                    )

            append_metrics_pair(
                metrics,
                metrics_by_level[abstraction_level],
                entry,
                ENTRY_USAGE_METRICS,
            )
            if focal_bucket is not None:
                append_metrics(
                    metrics_by_focal_bucket[focal_bucket],
                    entry,
                    ENTRY_USAGE_METRICS,
                )
            if focal_class_count_key is not None:
                append_metrics(
                    metrics_by_focal_class_count[focal_class_count_key],
                    entry,
                    ENTRY_USAGE_METRICS,
                )
            if focal_method_count_key is not None:
                append_metrics(
                    metrics_by_focal_method_count[focal_method_count_key],
                    entry,
                    ENTRY_USAGE_METRICS,
                )

            entry_cost = calculate_entry_cost(
                entry.input_tokens, entry.output_tokens, pricing_model
            )
            metrics["cost"].append(entry_cost)
            metrics_by_level[abstraction_level]["cost"].append(entry_cost)
            if focal_bucket is not None:
                metrics_by_focal_bucket[focal_bucket]["cost"].append(entry_cost)
            if focal_class_count_key is not None:
                metrics_by_focal_class_count[focal_class_count_key]["cost"].append(
                    entry_cost
                )
            if focal_method_count_key is not None:
                metrics_by_focal_method_count[focal_method_count_key]["cost"].append(
                    entry_cost
                )

            if not (has_structured and has_coverage and has_localization):
                problematic_entries.append(build_problem_entry(entry.nl2test_input))

    compile_rate = compiles_count / total_evals if total_evals else 0.0

    print_main_header("NL2Test Evaluation Summary")
    print(f"Output directory:    {output_dir}")
    print(f"Eval file:           {eval_file_name}")
    print(f"Pricing model:       {pricing_model}")
    print(f"Projects scanned:    {len(project_dirs)}")
    print(f"Projects with evals: {len(eval_files)}")
    if IGNORE_IF_NOT_COMPILE or ONLY_SHARED_ENTRIES:
        print(f"Skipped entries:     {len(skipped_ids)} (ignored IDs)")
    print(f"Total evaluations:   {total_evals}")
    print(f"Compiles:            {compiles_count} ({compile_rate:.1%})")
    print(f"Valid entries:       {valid_entries}")
    print(f"Problematic entries: {len(problematic_entries)}")
    print(f"Missing structured:  {missing_structured}")
    print(f"Missing coverage:    {missing_coverage}")
    print(f"Missing localization: {missing_localization}")
    print(f"Failed code generation:      {failed_code_generation_count}")
    print(f"Failed test file generation: {failed_test_file_generation_count}")
    print(f"Non-compiling IDs:   {len(non_compiling_ids)}")
    if non_compiling_ids:
        print(f"  IDs: {non_compiling_ids}")

    print("\nProjects with evaluation results:")
    for project_name in projects_with_evals:
        print(f"  - {project_name}")

    if count_by_level:
        print("\nEntries by abstraction level:")
        print(f"  {'Level':<10} {'Count':>8} {'Compiles':>10} {'Rate':>8}")
        print(f"  {'-' * 10} {'-' * 8} {'-' * 10} {'-' * 8}")
        for level in sort_abstraction_levels(count_by_level.keys()):
            count = count_by_level[level]
            compiles = compiles_by_level.get(level, 0)
            rate = compiles / count if count > 0 else 0.0
            print(f"  {level:<10} {count:>8} {compiles:>10} {rate:>7.1%}")

    if count_by_focal_bucket:
        print("\nEntries by focal method bucket:")
        print(f"  {'Bucket':<20} {'Count':>8} {'Compiles':>10} {'Rate':>8}")
        print(f"  {'-' * 20} {'-' * 8} {'-' * 10} {'-' * 8}")
        for bucket in sort_focal_buckets(count_by_focal_bucket.keys()):
            label = FOCAL_BUCKET_NAMES.get(bucket, bucket)
            count = count_by_focal_bucket[bucket]
            compiles = compiles_by_focal_bucket.get(bucket, 0)
            rate = compiles / count if count > 0 else 0.0
            print(f"  {label:<20} {count:>8} {compiles:>10} {rate:>7.1%}")

    if count_by_focal_class_count:
        print("\nEntries by focal class count:")
        print(f"  {'Focal Classes':<24} {'Count':>8} {'Compiles':>10} {'Rate':>8}")
        print(f"  {'-' * 24} {'-' * 8} {'-' * 10} {'-' * 8}")
        for count_key in sort_focal_class_counts(count_by_focal_class_count.keys()):
            count = count_by_focal_class_count[count_key]
            compiles = compiles_by_focal_class_count.get(count_key, 0)
            rate = compiles / count if count > 0 else 0.0
            print(f"  {count_key:<24} {count:>8} {compiles:>10} {rate:>7.1%}")

    if count_by_focal_method_count:
        print("\nEntries by focal method count (granular):")
        print(f"  {'Focal Methods':<24} {'Count':>8} {'Compiles':>10} {'Rate':>8}")
        print(f"  {'-' * 24} {'-' * 8} {'-' * 10} {'-' * 8}")
        for count_key in sort_focal_method_counts(count_by_focal_method_count.keys()):
            count = count_by_focal_method_count[count_key]
            compiles = compiles_by_focal_method_count.get(count_key, 0)
            rate = compiles / count if count > 0 else 0.0
            print(f"  {count_key:<24} {count:>8} {compiles:>10} {rate:>7.1%}")

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

    if metrics_by_focal_bucket:
        focal_buckets = sort_focal_buckets(metrics_by_focal_bucket.keys())
        per_focal_bucket_distributions = {
            bucket: build_distributions(bucket_metrics)
            for bucket, bucket_metrics in metrics_by_focal_bucket.items()
        }

        print_main_header("Distributions By Focal Method Bucket")
        print_category_by_level(
            "Structured Metrics",
            STRUCTURAL_METRICS,
            per_focal_bucket_distributions,
            focal_buckets,
        )
        print_category_by_level(
            "Coverage Metrics",
            COVERAGE_METRICS,
            per_focal_bucket_distributions,
            focal_buckets,
        )
        print_category_by_level(
            "Localization Metrics",
            LOCALIZATION_METRICS,
            per_focal_bucket_distributions,
            focal_buckets,
        )
        print_category_by_level(
            "Usage Metrics",
            USAGE_METRICS,
            per_focal_bucket_distributions,
            focal_buckets,
        )

    if metrics_by_focal_class_count:
        focal_class_counts = sort_focal_class_counts(
            metrics_by_focal_class_count.keys()
        )
        per_focal_class_distributions = {
            count_key: build_distributions(count_metrics)
            for count_key, count_metrics in metrics_by_focal_class_count.items()
        }

        print_main_header("Distributions By Focal Class Count")
        print_category_by_level(
            "Structured Metrics",
            STRUCTURAL_METRICS,
            per_focal_class_distributions,
            focal_class_counts,
        )
        print_category_by_level(
            "Coverage Metrics",
            COVERAGE_METRICS,
            per_focal_class_distributions,
            focal_class_counts,
        )
        print_category_by_level(
            "Localization Metrics",
            LOCALIZATION_METRICS,
            per_focal_class_distributions,
            focal_class_counts,
        )
        print_category_by_level(
            "Usage Metrics",
            USAGE_METRICS,
            per_focal_class_distributions,
            focal_class_counts,
        )

    if metrics_by_focal_method_count:
        focal_method_counts = sort_focal_method_counts(
            metrics_by_focal_method_count.keys()
        )
        per_focal_method_distributions = {
            count_key: build_distributions(count_metrics)
            for count_key, count_metrics in metrics_by_focal_method_count.items()
        }

        print_main_header("Distributions By Focal Method Count (Granular)")
        print_category_by_level(
            "Structured Metrics",
            STRUCTURAL_METRICS,
            per_focal_method_distributions,
            focal_method_counts,
        )
        print_category_by_level(
            "Coverage Metrics",
            COVERAGE_METRICS,
            per_focal_method_distributions,
            focal_method_counts,
        )
        print_category_by_level(
            "Localization Metrics",
            LOCALIZATION_METRICS,
            per_focal_method_distributions,
            focal_method_counts,
        )
        print_category_by_level(
            "Usage Metrics",
            USAGE_METRICS,
            per_focal_method_distributions,
            focal_method_counts,
        )

    json_output: Dict[str, Any] = {
        "summary": {
            "output_directory": str(output_dir.relative_to(ROOT_DIR)),
            "eval_file": eval_file_name,
            "pricing_model": pricing_model,
            "projects_scanned": len(project_dirs),
            "projects_with_evals": len(eval_files),
            "valid_entries": valid_entries,
            "problematic_entries": len(problematic_entries),
            "missing_structured": missing_structured,
            "missing_coverage": missing_coverage,
            "missing_localization": missing_localization,
            "failed_code_generation": failed_code_generation_count,
            "failed_test_file_generation": failed_test_file_generation_count,
            "non_compiling_ids": non_compiling_ids,
            "ignore_if_not_compile": IGNORE_IF_NOT_COMPILE,
            "only_shared_entries": ONLY_SHARED_ENTRIES,
            "ignore_ids": sorted(ignore_ids),
            "skipped_count": len(skipped_ids),
        },
        "holistic": {
            "count": total_evals,
            "compiles": compiles_count,
            "compile_rate": compile_rate,
            "distributions": distributions_to_dict(overall_distributions),
        },
    }

    if metrics_by_level:
        json_output["abstraction_levels"] = {
            level: {
                "count": count_by_level[level],
                "compiles": compiles_by_level.get(level, 0),
                "compile_rate": (
                    compiles_by_level.get(level, 0) / count_by_level[level]
                    if count_by_level[level] > 0
                    else 0.0
                ),
                "distributions": distributions_to_dict(per_level_distributions[level]),
            }
            for level in levels
        }

    if metrics_by_focal_bucket:
        sorted_buckets = sort_focal_buckets(metrics_by_focal_bucket.keys())
        bucket_distributions = {
            bucket: build_distributions(bucket_metrics)
            for bucket, bucket_metrics in metrics_by_focal_bucket.items()
        }
        json_output["focal_method_buckets"] = {
            bucket: {
                "count": count_by_focal_bucket[bucket],
                "compiles": compiles_by_focal_bucket.get(bucket, 0),
                "compile_rate": (
                    compiles_by_focal_bucket.get(bucket, 0)
                    / count_by_focal_bucket[bucket]
                    if count_by_focal_bucket[bucket] > 0
                    else 0.0
                ),
                "distributions": distributions_to_dict(bucket_distributions[bucket]),
            }
            for bucket in sorted_buckets
        }

    if metrics_by_focal_class_count:
        sorted_class_counts = sort_focal_class_counts(
            metrics_by_focal_class_count.keys()
        )
        class_count_distributions = {
            count_key: build_distributions(count_metrics)
            for count_key, count_metrics in metrics_by_focal_class_count.items()
        }
        json_output["focal_class_counts"] = {
            count_key: {
                "count": count_by_focal_class_count[count_key],
                "compiles": compiles_by_focal_class_count.get(count_key, 0),
                "compile_rate": (
                    compiles_by_focal_class_count.get(count_key, 0)
                    / count_by_focal_class_count[count_key]
                    if count_by_focal_class_count[count_key] > 0
                    else 0.0
                ),
                "distributions": distributions_to_dict(
                    class_count_distributions[count_key]
                ),
            }
            for count_key in sorted_class_counts
        }

    if metrics_by_focal_method_count:
        sorted_method_counts = sort_focal_method_counts(
            metrics_by_focal_method_count.keys()
        )
        method_count_distributions = {
            count_key: build_distributions(count_metrics)
            for count_key, count_metrics in metrics_by_focal_method_count.items()
        }
        json_output["focal_method_counts"] = {
            count_key: {
                "count": count_by_focal_method_count[count_key],
                "compiles": compiles_by_focal_method_count.get(count_key, 0),
                "compile_rate": (
                    compiles_by_focal_method_count.get(count_key, 0)
                    / count_by_focal_method_count[count_key]
                    if count_by_focal_method_count[count_key] > 0
                    else 0.0
                ),
                "distributions": distributions_to_dict(
                    method_count_distributions[count_key]
                ),
            }
            for count_key in sorted_method_counts
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_name = output_dir.name.replace("output", "eval") + ".json"
    output_file = OUTPUT_DIR / output_name
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(json_output, handle, indent=2)
    print(f"\nJSON results written to: {output_file}")


def main() -> None:
    args = parse_args()
    eval_file_name = args.eval_file_name

    ignore_ids: set[int] = set()
    if EVAL_DIRS:
        if IGNORE_IF_NOT_COMPILE:
            ignore_ids = build_ignore_ids_from_dirs(
                EVAL_DIRS, eval_file_name, require_compile=True
            )
        elif ONLY_SHARED_ENTRIES:
            ignore_ids = build_ignore_ids_from_dirs(
                EVAL_DIRS, eval_file_name, require_compile=False
            )

    for eval_dir in EVAL_DIRS:
        pricing_model = EVAL_DIR_PRICING[eval_dir]
        evaluate_directory(eval_dir, eval_file_name, pricing_model, ignore_ids)


if __name__ == "__main__":
    main()
