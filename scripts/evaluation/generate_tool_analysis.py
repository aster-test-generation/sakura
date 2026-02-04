"""Analyze tool usage patterns from NL2Test evaluation results."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import ValidationError

from sakura.dataset_creation.model import NL2TestDataset, Test
from sakura.utils.models.nl2test import OutOfBoxAgentEval
from sakura.utils.statistics import (
    distribution_to_dict,
    summarize_distribution,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
EVAL_FILE_NAME = "nl2test_evaluation_results.json"
OUTPUT_DIR = ROOT_DIR / "outputs" / "tool_analysis"
BUCKETED_FILTERED_DATASET_DIR = ROOT_DIR / "resources" / "filtered_bucketed_tests"
BUCKETED_DATASET_FILE = "nl2test.json"
RAW_OUTPUTS_DIR = ROOT_DIR / "outputs" / "raw_outputs"

EVAL_DIRS: List[Path] = [
    RAW_OUTPUTS_DIR / "gemini_cli_pro_output",
    RAW_OUTPUTS_DIR / "gemini_cli_flash_output",
    RAW_OUTPUTS_DIR / "nl2test_gemini_flash_output",
    RAW_OUTPUTS_DIR / "nl2test_gemini_pro_output",
    RAW_OUTPUTS_DIR / "nl2test_devstral_output",
    RAW_OUTPUTS_DIR / "nl2test_qwen3_output",
]

ABSTRACTION_ORDER = ("high", "medium", "low")
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

TOOL_CATEGORIES = {
    "retrieval": {
        "query_method_db",
        "query_class_db",
        "get_method_details",
        "search_reachable_methods_in_class",
        "read_file",
        "glob",
        "search_file_content",
        "list_directory",
        "extract_method_code",
        "get_inherited_library_classes",
    },
    "generation": {"generate_test_code", "write_file"},
    "validation": {"compile_and_execute_test", "run_shell_command"},
    "orchestration": {"call_localization_agent", "call_composition_agent", "finalize"},
    "inspection": {
        "view_test_code",
        "get_maven_dependencies",
        "get_class_fields",
        "get_class_constructors_and_factories",
        "get_getters_and_setters",
        "get_call_site_details",
        "modify_scenario_comment",
    },
}

QUALITY_METRICS = (
    "focal_recall",
    "focal_precision",
    "callable_recall",
    "callable_precision",
    "line_coverage",
    "branch_coverage",
    "localization_recall",
)

ONLY_SHARED_ENTRIES = True


@dataclass
class EntryToolData:
    entry_id: int
    project_name: str
    abstraction_level: str
    focal_bucket: Optional[str]

    tool_counts: Dict[str, int]
    tool_trajectories: List[List[str]]
    total_tool_calls: int
    agent_tool_counts: Dict[str, Dict[str, int]]
    agent_invocation_counts: Dict[str, int]

    compiles: bool
    focal_recall: Optional[float]
    focal_precision: Optional[float]
    callable_recall: Optional[float]
    callable_precision: Optional[float]
    line_coverage: Optional[float]
    branch_coverage: Optional[float]
    localization_recall: Optional[float]

    input_tokens: int
    output_tokens: int
    llm_calls: int


BucketTestMap = Dict[str, Tuple[Test, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze tool usage in NL2Test.")
    parser.add_argument(
        "--eval-file-name",
        type=str,
        default=EVAL_FILE_NAME,
        help="Evaluation file name inside each project directory.",
    )
    return parser.parse_args()


def normalize_abstraction_level(level: Any) -> str:
    if level is None:
        return "unknown"
    if isinstance(level, str):
        cleaned = level.strip()
        return cleaned if cleaned else "unknown"
    if hasattr(level, "value"):
        return str(level.value)
    return str(level)


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


def build_bucket_test_map(dataset: NL2TestDataset) -> BucketTestMap:
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


def build_shared_ids(dirs: List[Path], eval_file_name: str) -> set[int]:
    """Build set of IDs present in all directories."""
    ids_per_dir: List[set[int]] = []

    for eval_dir in dirs:
        if not eval_dir.exists():
            continue
        dir_ids: set[int] = set()
        project_dirs = [path for path in eval_dir.iterdir() if path.is_dir()]
        for project_dir in project_dirs:
            eval_file = project_dir / eval_file_name
            if not eval_file.is_file():
                continue
            entries = load_eval_file(eval_file)
            for entry in entries:
                dir_ids.add(entry.nl2test_input.id)
        ids_per_dir.append(dir_ids)

    if not ids_per_dir:
        return set()
    return set.intersection(*ids_per_dir)


def extract_tool_data(
    entry: OutOfBoxAgentEval, focal_bucket: Optional[str]
) -> EntryToolData:
    """Extract tool usage data from an evaluation entry."""
    tool_log = entry.tool_log
    tool_counts: Dict[str, int] = {}
    agent_tool_counts: Dict[str, Dict[str, int]] = {
        "supervisor": {},
        "localization": {},
        "composition": {},
    }
    agent_invocation_counts: Dict[str, int] = {
        "supervisor": 0,
        "localization": 0,
        "composition": 0,
    }
    all_trajectories: List[List[str]] = []

    if tool_log is not None:
        for tool_name, count in tool_log.supervisor_tool_log.tool_counts.items():
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + count
            agent_tool_counts["supervisor"][tool_name] = count
        for tool_name, count in tool_log.localization_tool_log.tool_counts.items():
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + count
            agent_tool_counts["localization"][tool_name] = count
        for tool_name, count in tool_log.composition_tool_log.tool_counts.items():
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + count
            agent_tool_counts["composition"][tool_name] = count

        all_trajectories.extend(tool_log.supervisor_tool_log.tool_trajectories)
        all_trajectories.extend(tool_log.localization_tool_log.tool_trajectories)
        all_trajectories.extend(tool_log.composition_tool_log.tool_trajectories)

        agent_invocation_counts["supervisor"] = len(
            tool_log.supervisor_tool_log.tool_trajectories
        )
        agent_invocation_counts["localization"] = len(
            tool_log.localization_tool_log.tool_trajectories
        )
        agent_invocation_counts["composition"] = len(
            tool_log.composition_tool_log.tool_trajectories
        )

    total_calls = sum(tool_counts.values())

    focal_recall: Optional[float] = None
    focal_precision: Optional[float] = None
    callable_recall: Optional[float] = None
    callable_precision: Optional[float] = None
    line_coverage: Optional[float] = None
    branch_coverage: Optional[float] = None
    localization_recall: Optional[float] = None

    if entry.structured_eval is not None:
        focal_recall = entry.structured_eval.focal_recall
        focal_precision = entry.structured_eval.focal_precision
        callable_recall = entry.structured_eval.callable_recall
        callable_precision = entry.structured_eval.callable_precision

    if entry.coverage_eval is not None:
        line_coverage = entry.coverage_eval.line_coverage
        branch_coverage = entry.coverage_eval.branch_coverage

    if entry.localization_eval is not None:
        if hasattr(entry.localization_eval, "localization_recall"):
            localization_recall = entry.localization_eval.localization_recall
        elif isinstance(entry.localization_eval, dict):
            localization_recall = entry.localization_eval.get("localization_recall")

    return EntryToolData(
        entry_id=entry.nl2test_input.id,
        project_name=entry.nl2test_input.project_name,
        abstraction_level=normalize_abstraction_level(
            entry.nl2test_input.abstraction_level
        ),
        focal_bucket=focal_bucket,
        tool_counts=tool_counts,
        tool_trajectories=all_trajectories,
        total_tool_calls=total_calls,
        agent_tool_counts=agent_tool_counts,
        agent_invocation_counts=agent_invocation_counts,
        compiles=entry.compiles,
        focal_recall=focal_recall,
        focal_precision=focal_precision,
        callable_recall=callable_recall,
        callable_precision=callable_precision,
        line_coverage=line_coverage,
        branch_coverage=branch_coverage,
        localization_recall=localization_recall,
        input_tokens=entry.input_tokens,
        output_tokens=entry.output_tokens,
        llm_calls=entry.llm_calls,
    )


def load_all_entries(
    eval_dir: Path,
    eval_file_name: str,
    bucketed_datasets: Dict[str, NL2TestDataset | None],
    bucket_test_maps: Dict[str, BucketTestMap],
    valid_ids: set[int] | None,
) -> List[EntryToolData]:
    """Load all evaluation entries from an evaluation directory."""
    entries: List[EntryToolData] = []

    if not eval_dir.exists():
        return entries

    project_dirs = [path for path in eval_dir.iterdir() if path.is_dir()]

    for project_dir in sorted(project_dirs, key=lambda p: p.name):
        eval_file = project_dir / eval_file_name
        if not eval_file.is_file():
            continue

        project_name = project_dir.name
        if project_name not in bucketed_datasets:
            bucketed_datasets[project_name] = load_bucketed_dataset(project_name)
            dataset = bucketed_datasets[project_name]
            if dataset is not None:
                bucket_test_maps[project_name] = build_bucket_test_map(dataset)

        eval_entries = load_eval_file(eval_file)
        for entry in eval_entries:
            entry_id = entry.nl2test_input.id
            if valid_ids is not None and entry_id not in valid_ids:
                continue

            focal_bucket: Optional[str] = None
            test_key = f"{entry.nl2test_input.qualified_class_name}::{entry.nl2test_input.method_signature}"
            if project_name in bucket_test_maps:
                bucket_map = bucket_test_maps[project_name]
                if test_key in bucket_map:
                    _, focal_bucket = bucket_map[test_key]

            entry_data = extract_tool_data(entry, focal_bucket)
            entries.append(entry_data)

    return entries


def categorize_tool(tool_name: str) -> str:
    """Return the category of a tool."""
    for category, tools in TOOL_CATEGORIES.items():
        if tool_name in tools:
            return category
    return "other"


def calculate_tool_diversity(tool_counts: Dict[str, int]) -> float:
    """Calculate Shannon entropy as a measure of tool diversity."""
    total = sum(tool_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in tool_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def find_common_ngrams(
    trajectories: List[List[str]], n: int, top_k: int
) -> List[Tuple[Tuple[str, ...], int]]:
    """Find the most common n-grams in tool trajectories."""
    ngram_counts: Counter[Tuple[str, ...]] = Counter()
    for trajectory in trajectories:
        if len(trajectory) < n:
            continue
        for i in range(len(trajectory) - n + 1):
            ngram = tuple(trajectory[i : i + n])
            ngram_counts[ngram] += 1
    return ngram_counts.most_common(top_k)


def detect_retry_patterns(trajectories: List[List[str]]) -> Dict[str, int]:
    """Detect consecutive repeated tool calls (retries)."""
    retry_counts: Counter[str] = Counter()
    for trajectory in trajectories:
        prev_tool: Optional[str] = None
        for tool in trajectory:
            if tool == prev_tool:
                retry_counts[tool] += 1
            prev_tool = tool
    return dict(retry_counts)


def analyze_tool_efficiency(entries: List[EntryToolData]) -> Dict[str, Any]:
    """Analyze correlation between tool calls and quality outcomes."""
    compiling_entries = [e for e in entries if e.compiles]
    non_compiling_entries = [e for e in entries if not e.compiles]

    compiling_calls = [float(e.total_tool_calls) for e in compiling_entries]
    non_compiling_calls = [float(e.total_tool_calls) for e in non_compiling_entries]

    compiling_dist = summarize_distribution(compiling_calls)
    non_compiling_dist = summarize_distribution(non_compiling_calls)

    # Bin entries by tool call count
    bins = [(0, 10), (11, 25), (26, 50), (51, 100), (101, float("inf"))]
    bin_labels = ["0-10", "11-25", "26-50", "51-100", "100+"]
    bin_data: Dict[str, Dict[str, Any]] = {}

    for (low, high), label in zip(bins, bin_labels):
        bin_entries = [e for e in entries if low <= e.total_tool_calls <= high]
        if not bin_entries:
            continue

        compile_rate = sum(1 for e in bin_entries if e.compiles) / len(bin_entries)
        focal_recalls = [
            e.focal_recall for e in bin_entries if e.focal_recall is not None
        ]
        line_coverages = [
            e.line_coverage for e in bin_entries if e.line_coverage is not None
        ]

        bin_data[label] = {
            "count": len(bin_entries),
            "compile_rate": compile_rate,
            "focal_recall": (
                distribution_to_dict(summarize_distribution(focal_recalls))
                if focal_recalls
                else None
            ),
            "line_coverage": (
                distribution_to_dict(summarize_distribution(line_coverages))
                if line_coverages
                else None
            ),
        }

    # Quality per tool call
    quality_per_call: Dict[str, Any] = {}
    for metric in ["focal_recall", "line_coverage"]:
        values = []
        for e in entries:
            metric_val = getattr(e, metric)
            if metric_val is not None and e.total_tool_calls > 0:
                values.append(metric_val / e.total_tool_calls)
        if values:
            quality_per_call[metric] = distribution_to_dict(
                summarize_distribution(values)
            )

    return {
        "compiling_tool_calls": distribution_to_dict(compiling_dist),
        "non_compiling_tool_calls": distribution_to_dict(non_compiling_dist),
        "tool_call_bins": bin_data,
        "quality_per_call": quality_per_call,
    }


def analyze_tool_categories(entries: List[EntryToolData]) -> Dict[str, Any]:
    """Analyze tool usage by category."""
    category_counts: Dict[str, int] = {cat: 0 for cat in TOOL_CATEGORIES}
    category_counts["other"] = 0

    agent_category_counts: Dict[str, Dict[str, int]] = {
        agent: {cat: 0 for cat in list(TOOL_CATEGORIES.keys()) + ["other"]}
        for agent in ["supervisor", "localization", "composition"]
    }

    for entry in entries:
        for tool_name, count in entry.tool_counts.items():
            category = categorize_tool(tool_name)
            category_counts[category] += count

        for agent, agent_tools in entry.agent_tool_counts.items():
            for tool_name, count in agent_tools.items():
                category = categorize_tool(tool_name)
                agent_category_counts[agent][category] += count

    total_calls = sum(category_counts.values())
    category_proportions = (
        {cat: count / total_calls for cat, count in category_counts.items()}
        if total_calls > 0
        else {}
    )

    retrieval = category_counts.get("retrieval", 0)
    generation = category_counts.get("generation", 0)
    read_write_ratio = retrieval / generation if generation > 0 else float("inf")

    return {
        "category_counts": category_counts,
        "category_proportions": category_proportions,
        "read_write_ratio": read_write_ratio,
        "agent_category_counts": agent_category_counts,
        "total_tool_calls": total_calls,
    }


def analyze_trajectories(entries: List[EntryToolData]) -> Dict[str, Any]:
    """Analyze tool trajectory patterns."""
    all_trajectories: List[List[str]] = []
    trajectory_lengths: List[float] = []
    invocation_counts: List[float] = []

    for entry in entries:
        all_trajectories.extend(entry.tool_trajectories)
        for traj in entry.tool_trajectories:
            trajectory_lengths.append(float(len(traj)))
        total_invocations = sum(entry.agent_invocation_counts.values())
        invocation_counts.append(float(total_invocations))

    bigrams = find_common_ngrams(all_trajectories, 2, 15)
    trigrams = find_common_ngrams(all_trajectories, 3, 10)
    retry_patterns = detect_retry_patterns(all_trajectories)

    return {
        "trajectory_length_distribution": distribution_to_dict(
            summarize_distribution(trajectory_lengths)
        ),
        "invocation_count_distribution": distribution_to_dict(
            summarize_distribution(invocation_counts)
        ),
        "common_bigrams": [
            {"pattern": list(pattern), "count": count} for pattern, count in bigrams
        ],
        "common_trigrams": [
            {"pattern": list(pattern), "count": count} for pattern, count in trigrams
        ],
        "retry_patterns": retry_patterns,
        "total_trajectories": len(all_trajectories),
    }


def analyze_by_dimension(
    entries: List[EntryToolData],
    dimension_getter: Any,
    sort_fn: Any,
) -> Dict[str, Any]:
    """Analyze tool usage by a specific dimension."""
    by_dimension: Dict[str, List[EntryToolData]] = {}
    for entry in entries:
        dim_val = dimension_getter(entry)
        if dim_val is None:
            continue
        if dim_val not in by_dimension:
            by_dimension[dim_val] = []
        by_dimension[dim_val].append(entry)

    result: Dict[str, Any] = {}
    sorted_dims = sort_fn(by_dimension.keys())

    for dim in sorted_dims:
        dim_entries = by_dimension[dim]
        tool_calls = [float(e.total_tool_calls) for e in dim_entries]
        compile_rate = (
            sum(1 for e in dim_entries if e.compiles) / len(dim_entries)
            if dim_entries
            else 0
        )

        category_counts: Dict[str, int] = {cat: 0 for cat in TOOL_CATEGORIES}
        category_counts["other"] = 0
        for entry in dim_entries:
            for tool_name, count in entry.tool_counts.items():
                category = categorize_tool(tool_name)
                category_counts[category] += count

        total = sum(category_counts.values())
        category_proportions = (
            {cat: count / total for cat, count in category_counts.items()}
            if total > 0
            else {}
        )

        focal_recalls = [
            e.focal_recall for e in dim_entries if e.focal_recall is not None
        ]

        result[dim] = {
            "count": len(dim_entries),
            "compile_rate": compile_rate,
            "tool_calls": distribution_to_dict(summarize_distribution(tool_calls)),
            "category_counts": category_counts,
            "category_proportions": category_proportions,
            "focal_recall": (
                distribution_to_dict(summarize_distribution(focal_recalls))
                if focal_recalls
                else None
            ),
        }

    return result


def build_cross_system_comparison(
    all_analyses: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build comparison across different systems."""
    comparison: Dict[str, Any] = {}

    for dir_name, analysis in all_analyses.items():
        if "tool_categories" not in analysis:
            continue

        categories = analysis["tool_categories"]
        entry_count = analysis.get("entry_count", 1)

        diversity = calculate_tool_diversity(categories.get("category_counts", {}))
        total_calls = categories.get("total_tool_calls", 0)
        calls_per_entry = total_calls / entry_count if entry_count > 0 else 0

        compile_rate = analysis.get("compile_rate", 0)
        efficiency = analysis.get("tool_efficiency", {})
        quality_per_call = efficiency.get("quality_per_call", {})

        comparison[dir_name] = {
            "entry_count": entry_count,
            "total_tool_calls": total_calls,
            "calls_per_entry": calls_per_entry,
            "tool_diversity_index": diversity,
            "compile_rate": compile_rate,
            "category_proportions": categories.get("category_proportions", {}),
            "read_write_ratio": categories.get("read_write_ratio", 0),
            "quality_per_call": quality_per_call,
        }

    return comparison


def print_main_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_category_header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def print_tool_efficiency_summary(efficiency: Dict[str, Any]) -> None:
    print_category_header("Tool Calls by Compile Status")

    compiling = efficiency.get("compiling_tool_calls", {})
    non_compiling = efficiency.get("non_compiling_tool_calls", {})

    print(f"  {'Status':<15} {'Mean':>8} {'P50':>8} {'P75':>8} {'P90':>8} {'Count':>8}")
    print(f"  {'-' * 15} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    if compiling:
        print(
            f"  {'Compiling':<15} {compiling.get('mean', 0):>8.1f} {compiling.get('p50', 0):>8.1f} "
            f"{compiling.get('p75', 0):>8.1f} {compiling.get('p90', 0):>8.1f} {compiling.get('count', 0):>8}"
        )
    if non_compiling:
        print(
            f"  {'Non-compiling':<15} {non_compiling.get('mean', 0):>8.1f} {non_compiling.get('p50', 0):>8.1f} "
            f"{non_compiling.get('p75', 0):>8.1f} {non_compiling.get('p90', 0):>8.1f} {non_compiling.get('count', 0):>8}"
        )

    print_category_header("Quality by Tool Call Bins")
    bins = efficiency.get("tool_call_bins", {})
    if bins:
        print(
            f"  {'Bin':<12} {'Count':>8} {'Compile%':>10} {'FocalRecall':>12} {'LineCov':>12}"
        )
        print(f"  {'-' * 12} {'-' * 8} {'-' * 10} {'-' * 12} {'-' * 12}")
        for bin_label, data in bins.items():
            focal = data.get("focal_recall", {})
            line = data.get("line_coverage", {})
            focal_mean = focal.get("mean", 0) if focal else 0
            line_mean = line.get("mean", 0) if line else 0
            print(
                f"  {bin_label:<12} {data.get('count', 0):>8} {data.get('compile_rate', 0) * 100:>9.1f}% "
                f"{focal_mean:>11.3f} {line_mean:>11.3f}"
            )


def print_category_summary(categories: Dict[str, Any]) -> None:
    print_category_header("Tool Category Distribution")

    counts = categories.get("category_counts", {})
    proportions = categories.get("category_proportions", {})
    total = categories.get("total_tool_calls", 0)

    print(f"  {'Category':<15} {'Count':>10} {'Proportion':>12}")
    print(f"  {'-' * 15} {'-' * 10} {'-' * 12}")
    for cat in list(TOOL_CATEGORIES.keys()) + ["other"]:
        count = counts.get(cat, 0)
        prop = proportions.get(cat, 0)
        print(f"  {cat:<15} {count:>10} {prop * 100:>11.1f}%")
    print(f"  {'TOTAL':<15} {total:>10}")

    print(f"\n  Read/Write Ratio: {categories.get('read_write_ratio', 0):.2f}")


def print_trajectory_summary(trajectories: Dict[str, Any]) -> None:
    print_category_header("Trajectory Analysis")

    length_dist = trajectories.get("trajectory_length_distribution", {})
    print(
        f"  Trajectory lengths: mean={length_dist.get('mean', 0):.1f}, "
        f"p50={length_dist.get('p50', 0):.1f}, max={length_dist.get('max', 0):.0f}"
    )
    print(f"  Total trajectories: {trajectories.get('total_trajectories', 0)}")

    print("\n  Top Bigrams:")
    for item in trajectories.get("common_bigrams", [])[:10]:
        print(f"    {' -> '.join(item['pattern'])}: {item['count']}")

    print("\n  Retry Patterns (consecutive repeated calls):")
    retries = trajectories.get("retry_patterns", {})
    if retries:
        sorted_retries = sorted(retries.items(), key=lambda x: x[1], reverse=True)[:10]
        for tool, count in sorted_retries:
            print(f"    {tool}: {count}")
    else:
        print("    None detected")


def print_dimension_summary(
    title: str, data: Dict[str, Any], dim_names: Dict[str, str] | None = None
) -> None:
    print_category_header(title)

    print(
        f"  {'Dimension':<24} {'Count':>8} {'Compile%':>10} {'ToolCalls':>12} {'FocalRec':>10}"
    )
    print(f"  {'-' * 24} {'-' * 8} {'-' * 10} {'-' * 12} {'-' * 10}")

    for dim, dim_data in data.items():
        label = dim_names.get(dim, dim) if dim_names else dim
        tool_calls = dim_data.get("tool_calls", {})
        focal = dim_data.get("focal_recall", {})
        print(
            f"  {label:<24} {dim_data.get('count', 0):>8} "
            f"{dim_data.get('compile_rate', 0) * 100:>9.1f}% "
            f"{tool_calls.get('mean', 0):>11.1f} "
            f"{focal.get('mean', 0) if focal else 0:>9.3f}"
        )


def evaluate_directory(
    output_dir: Path,
    eval_file_name: str,
    valid_ids: set[int] | None,
    bucketed_datasets: Dict[str, NL2TestDataset | None],
    bucket_test_maps: Dict[str, BucketTestMap],
) -> Dict[str, Any]:
    """Process one directory and generate tool analysis."""
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return {}

    entries = load_all_entries(
        output_dir, eval_file_name, bucketed_datasets, bucket_test_maps, valid_ids
    )

    if not entries:
        print(f"No entries found in {output_dir}")
        return {}

    compile_rate = sum(1 for e in entries if e.compiles) / len(entries)

    tool_efficiency = analyze_tool_efficiency(entries)
    tool_categories = analyze_tool_categories(entries)
    tool_trajectories = analyze_trajectories(entries)

    tool_by_abstraction = analyze_by_dimension(
        entries, lambda e: e.abstraction_level, sort_abstraction_levels
    )

    tool_by_focal_bucket = analyze_by_dimension(
        entries, lambda e: e.focal_bucket, sort_focal_buckets
    )

    print_main_header(f"Tool Analysis: {output_dir.name}")
    print(f"Total entries: {len(entries)}")
    print(f"Compile rate: {compile_rate:.1%}")

    print_tool_efficiency_summary(tool_efficiency)
    print_category_summary(tool_categories)
    print_trajectory_summary(tool_trajectories)
    print_dimension_summary("Tool Usage by Abstraction Level", tool_by_abstraction)
    print_dimension_summary(
        "Tool Usage by Focal Bucket", tool_by_focal_bucket, FOCAL_BUCKET_NAMES
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dir_name = output_dir.name.replace("output", "").strip("_")

    analysis_result = {
        "summary": {
            "output_directory": str(output_dir.relative_to(ROOT_DIR)),
            "eval_file": eval_file_name,
            "entry_count": len(entries),
            "compile_rate": compile_rate,
        },
        "entry_count": len(entries),
        "compile_rate": compile_rate,
        "tool_efficiency": tool_efficiency,
        "tool_categories": tool_categories,
        "tool_trajectories": tool_trajectories,
        "tool_by_abstraction": tool_by_abstraction,
        "tool_by_focal_bucket": tool_by_focal_bucket,
    }

    output_file = OUTPUT_DIR / f"{dir_name}_tool_analysis.json"
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(analysis_result, handle, indent=2)
    print(f"\nJSON results written to: {output_file}")

    return analysis_result


def main() -> None:
    args = parse_args()
    eval_file_name = args.eval_file_name

    valid_ids: set[int] | None = None
    if ONLY_SHARED_ENTRIES and EVAL_DIRS:
        valid_ids = build_shared_ids(EVAL_DIRS, eval_file_name)
        print(f"\nUsing shared entries only: {len(valid_ids)} IDs")

    bucketed_datasets: Dict[str, NL2TestDataset | None] = {}
    bucket_test_maps: Dict[str, BucketTestMap] = {}

    all_analyses: Dict[str, Dict[str, Any]] = {}

    for eval_dir in EVAL_DIRS:
        if not eval_dir.exists():
            print(f"Skipping non-existent directory: {eval_dir}")
            continue

        analysis = evaluate_directory(
            eval_dir,
            eval_file_name,
            valid_ids,
            bucketed_datasets,
            bucket_test_maps,
        )
        if analysis:
            all_analyses[eval_dir.name] = analysis

    if len(all_analyses) > 1:
        comparison = build_cross_system_comparison(all_analyses)

        print_main_header("Cross-System Comparison")
        print(
            f"  {'System':<35} {'Entries':>8} {'Calls/Entry':>12} {'Diversity':>10} {'Compile%':>10}"
        )
        print(f"  {'-' * 35} {'-' * 8} {'-' * 12} {'-' * 10} {'-' * 10}")
        for system, data in comparison.items():
            print(
                f"  {system:<35} {data.get('entry_count', 0):>8} "
                f"{data.get('calls_per_entry', 0):>11.1f} "
                f"{data.get('tool_diversity_index', 0):>9.2f} "
                f"{data.get('compile_rate', 0) * 100:>9.1f}%"
            )

        comparison_file = OUTPUT_DIR / "tool_comparison.json"
        with comparison_file.open("w", encoding="utf-8") as handle:
            json.dump(comparison, handle, indent=2)
        print(f"\nComparison results written to: {comparison_file}")


if __name__ == "__main__":
    main()
