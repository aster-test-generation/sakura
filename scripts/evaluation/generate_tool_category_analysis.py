"""Analyze tool usage by functional categories across NL2Test and Gemini CLI."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from sakura.dataset_creation.model import NL2TestDataset, Test
from sakura.utils.models.nl2test import OutOfBoxAgentEval
from sakura.utils.statistics import distribution_to_dict, summarize_distribution

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

ONLY_SHARED_ENTRIES = True

# Category 1: Retrieval (Semantic Search)
RETRIEVAL_TOOLS = {
    "query_method_db",
    "query_class_db",
    "search_reachable_methods_in_class",
    "glob",
    "search_file_content",
}

# Category 2: Inspection (Static Code Reading)
INSPECTION_TOOLS = {
    "extract_method_code",
    "get_method_details",
    "get_call_site_details",
    "get_inherited_library_classes",
    "get_class_fields",
    "get_class_constructors_and_factories",
    "get_getters_and_setters",
    "get_maven_dependencies",
    "view_test_code",
    "read_file",
    "list_directory",
}

# Category 3: Generation (Code Writing)
GENERATION_TOOLS = {
    "generate_test_code",
    "modify_scenario_comment",
    "write_file",
    "replace",
}

# Category 4: Validation (Compile/Execute)
VALIDATION_TOOLS = {
    "compile_and_execute_test",
    "run_shell_command",
}

# Category 5: Orchestration (Agent Coordination)
ORCHESTRATION_TOOLS = {
    "call_localization_agent",
    "call_composition_agent",
    "finalize",
    "delegate_to_agent",
}

# Category 6: External (Other)
EXTERNAL_TOOLS = {
    "google_web_search",
    "write_todos",
}

TOOL_CATEGORIES = {
    "retrieval": RETRIEVAL_TOOLS,
    "inspection": INSPECTION_TOOLS,
    "generation": GENERATION_TOOLS,
    "validation": VALIDATION_TOOLS,
    "orchestration": ORCHESTRATION_TOOLS,
    "external": EXTERNAL_TOOLS,
}

CATEGORY_ORDER = [
    "retrieval",
    "inspection",
    "generation",
    "validation",
    "orchestration",
    "external",
    "other",
]

CATEGORY_DISPLAY_NAMES = {
    "retrieval": "Retrieval",
    "inspection": "Inspection",
    "generation": "Generation",
    "validation": "Validation",
    "orchestration": "Orchestration",
    "external": "External",
    "other": "Other",
}

# Tool to agent mapping for NL2Test tools
TOOL_TO_AGENTS: Dict[str, List[str]] = {
    "query_method_db": ["localization"],
    "query_class_db": ["localization", "composition"],
    "search_reachable_methods_in_class": ["localization"],
    "extract_method_code": ["localization", "composition"],
    "get_method_details": ["localization", "composition"],
    "get_call_site_details": ["localization", "composition"],
    "get_inherited_library_classes": ["localization"],
    "get_class_fields": ["composition"],
    "get_class_constructors_and_factories": ["composition"],
    "get_getters_and_setters": ["composition"],
    "get_maven_dependencies": ["composition"],
    "view_test_code": ["supervisor", "composition"],
    "generate_test_code": ["composition"],
    "modify_scenario_comment": ["composition"],
    "compile_and_execute_test": ["supervisor", "composition"],
    "call_localization_agent": ["supervisor"],
    "call_composition_agent": ["supervisor"],
    "finalize": ["supervisor", "localization", "composition"],
}


@dataclass
class EntryToolData:
    entry_id: int
    project_name: str
    tool_counts: Dict[str, int]
    agent_tool_counts: Dict[str, Dict[str, int]]
    total_tool_calls: int
    compiles: bool
    focal_recall: Optional[float]
    focal_precision: Optional[float]
    line_coverage: Optional[float]
    branch_coverage: Optional[float]


BucketTestMap = Dict[str, Tuple[Test, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze tool usage by functional categories."
    )
    parser.add_argument(
        "--eval-file-name",
        type=str,
        default=EVAL_FILE_NAME,
        help="Evaluation file name inside each project directory.",
    )
    return parser.parse_args()


def categorize_tool(tool_name: str) -> str:
    for category, tools in TOOL_CATEGORIES.items():
        if tool_name in tools:
            return category
    return "other"


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


def extract_tool_data(entry: OutOfBoxAgentEval) -> EntryToolData:
    tool_log = entry.tool_log
    tool_counts: Dict[str, int] = {}
    agent_tool_counts: Dict[str, Dict[str, int]] = {
        "supervisor": {},
        "localization": {},
        "composition": {},
    }

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

    total_calls = sum(tool_counts.values())

    focal_recall: Optional[float] = None
    focal_precision: Optional[float] = None
    line_coverage: Optional[float] = None
    branch_coverage: Optional[float] = None

    if entry.structured_eval is not None:
        focal_recall = entry.structured_eval.focal_recall
        focal_precision = entry.structured_eval.focal_precision

    if entry.coverage_eval is not None:
        line_coverage = entry.coverage_eval.line_coverage
        branch_coverage = entry.coverage_eval.branch_coverage

    return EntryToolData(
        entry_id=entry.nl2test_input.id,
        project_name=entry.nl2test_input.project_name,
        tool_counts=tool_counts,
        agent_tool_counts=agent_tool_counts,
        total_tool_calls=total_calls,
        compiles=entry.compiles,
        focal_recall=focal_recall,
        focal_precision=focal_precision,
        line_coverage=line_coverage,
        branch_coverage=branch_coverage,
    )


def load_all_entries(
    eval_dir: Path,
    eval_file_name: str,
    bucketed_datasets: Dict[str, NL2TestDataset | None],
    bucket_test_maps: Dict[str, BucketTestMap],
    valid_ids: set[int] | None,
) -> List[EntryToolData]:
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

            entry_data = extract_tool_data(entry)
            entries.append(entry_data)

    return entries


def compute_category_totals(entries: List[EntryToolData]) -> Dict[str, Dict[str, Any]]:
    category_counts: Counter[str] = Counter()

    for entry in entries:
        for tool_name, count in entry.tool_counts.items():
            category = categorize_tool(tool_name)
            category_counts[category] += count

    total = sum(category_counts.values())

    result: Dict[str, Dict[str, Any]] = {}
    for category in CATEGORY_ORDER:
        count = category_counts.get(category, 0)
        proportion = count / total if total > 0 else 0.0
        result[category] = {"count": count, "proportion": proportion}

    return result


def compute_agent_categories(
    entries: List[EntryToolData],
) -> Dict[str, Dict[str, int]]:
    agent_categories: Dict[str, Dict[str, int]] = {
        "supervisor": {cat: 0 for cat in CATEGORY_ORDER},
        "localization": {cat: 0 for cat in CATEGORY_ORDER},
        "composition": {cat: 0 for cat in CATEGORY_ORDER},
    }

    for entry in entries:
        for agent, agent_tools in entry.agent_tool_counts.items():
            for tool_name, count in agent_tools.items():
                category = categorize_tool(tool_name)
                agent_categories[agent][category] += count

    return agent_categories


def compute_tool_details(
    entries: List[EntryToolData],
) -> Dict[str, Dict[str, Any]]:
    tool_counts: Counter[str] = Counter()
    tool_agents: Dict[str, set[str]] = {}

    for entry in entries:
        for tool_name, count in entry.tool_counts.items():
            tool_counts[tool_name] += count

        for agent, agent_tools in entry.agent_tool_counts.items():
            for tool_name in agent_tools:
                if tool_name not in tool_agents:
                    tool_agents[tool_name] = set()
                tool_agents[tool_name].add(agent)

    result: Dict[str, Dict[str, Any]] = {}
    for tool_name, count in tool_counts.most_common():
        category = categorize_tool(tool_name)
        agents = sorted(tool_agents.get(tool_name, set()))
        result[tool_name] = {
            "count": count,
            "category": category,
            "agents": agents,
        }

    return result


def compute_ratios(category_totals: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    retrieval = category_totals.get("retrieval", {}).get("count", 0)
    inspection = category_totals.get("inspection", {}).get("count", 0)
    generation = category_totals.get("generation", {}).get("count", 0)
    validation = category_totals.get("validation", {}).get("count", 0)

    def safe_ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator > 0 else float("inf")

    return {
        "retrieval_to_generation": safe_ratio(retrieval, generation),
        "inspection_to_generation": safe_ratio(inspection, generation),
        "validation_to_generation": safe_ratio(validation, generation),
    }


def compute_quality_by_dominant_category(
    entries: List[EntryToolData],
) -> Dict[str, Dict[str, Any]]:
    category_entries: Dict[str, List[EntryToolData]] = {
        cat: [] for cat in CATEGORY_ORDER
    }

    for entry in entries:
        if entry.total_tool_calls == 0:
            continue

        category_counts: Counter[str] = Counter()
        for tool_name, count in entry.tool_counts.items():
            category = categorize_tool(tool_name)
            category_counts[category] += count

        dominant = category_counts.most_common(1)[0][0]
        category_entries[dominant].append(entry)

    result: Dict[str, Dict[str, Any]] = {}
    for category in CATEGORY_ORDER:
        cat_entries = category_entries[category]
        if not cat_entries:
            continue

        compile_rate = sum(1 for e in cat_entries if e.compiles) / len(cat_entries)
        focal_recalls = [
            e.focal_recall for e in cat_entries if e.focal_recall is not None
        ]
        line_coverages = [
            e.line_coverage for e in cat_entries if e.line_coverage is not None
        ]

        result[category] = {
            "count": len(cat_entries),
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

    return result


def is_nl2test_dir(dir_name: str) -> bool:
    return "nl2test" in dir_name.lower()


def print_main_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_section_header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def print_category_distribution(category_totals: Dict[str, Dict[str, Any]]) -> None:
    print_section_header("Category Distribution (All Tools)")

    print(f"  {'Category':<19} {'Count':>8} {'Proportion':>12}")
    print(f"  {'-' * 19} {'-' * 8} {'-' * 12}")

    total = 0
    for category in CATEGORY_ORDER:
        data = category_totals.get(category, {"count": 0, "proportion": 0})
        count = data["count"]
        proportion = data["proportion"]
        total += count
        display_name = CATEGORY_DISPLAY_NAMES.get(category, category)
        print(f"  {display_name:<19} {count:>8} {proportion * 100:>11.1f}%")

    print(f"  {'-' * 19} {'-' * 8} {'-' * 12}")
    print(f"  {'TOTAL':<19} {total:>8}")


def print_agent_category_distribution(
    agent_categories: Dict[str, Dict[str, int]],
) -> None:
    print_section_header("Per-Agent Category Distribution (NL2Test)")

    headers = ["Retrieval", "Inspection", "Generation", "Validation", "Orch."]
    cats = [
        "retrieval",
        "inspection",
        "generation",
        "validation",
        "orchestration",
    ]

    print(f"  {'Agent':<15}", end="")
    for header in headers:
        print(f" {header:>10}", end="")
    print()
    print(f"  {'-' * 15}", end="")
    for _ in headers:
        print(f" {'-' * 10}", end="")
    print()

    for agent in ["supervisor", "localization", "composition"]:
        agent_data = agent_categories.get(agent, {})
        total = sum(agent_data.values())
        print(f"  {agent.capitalize():<15}", end="")
        for cat in cats:
            count = agent_data.get(cat, 0)
            pct = count / total * 100 if total > 0 else 0.0
            print(f" {pct:>9.1f}%", end="")
        print()


def print_ratios(ratios: Dict[str, float]) -> None:
    print_section_header("Key Ratios")

    ret_gen = ratios.get("retrieval_to_generation", 0)
    insp_gen = ratios.get("inspection_to_generation", 0)
    val_gen = ratios.get("validation_to_generation", 0)

    ret_str = f"{ret_gen:.2f}" if ret_gen != float("inf") else "inf"
    insp_str = f"{insp_gen:.2f}" if insp_gen != float("inf") else "inf"
    val_str = f"{val_gen:.2f}" if val_gen != float("inf") else "inf"

    print(f"  Retrieval/Generation Ratio: {ret_str}")
    print(f"  Inspection/Generation Ratio: {insp_str}")
    print(f"  Validation/Generation Ratio: {val_str}")


def print_tool_details(
    tool_details: Dict[str, Dict[str, Any]], top_n: int = 15
) -> None:
    print_section_header(f"Top {top_n} Tools by Usage")

    print(f"  {'Tool':<40} {'Count':>8} {'Category':<16} {'Agents':<30}")
    print(f"  {'-' * 40} {'-' * 8} {'-' * 16} {'-' * 30}")

    for i, (tool_name, data) in enumerate(tool_details.items()):
        if i >= top_n:
            break
        count = data["count"]
        category = CATEGORY_DISPLAY_NAMES.get(data["category"], data["category"])
        agents = ", ".join(data["agents"]) if data["agents"] else "-"
        print(f"  {tool_name:<40} {count:>8} {category:<16} {agents:<30}")


def print_quality_by_dominant_category(
    quality_data: Dict[str, Dict[str, Any]],
) -> None:
    print_section_header("Quality by Dominant Category")

    print(
        f"  {'Category':<19} {'Count':>8} {'Compile%':>10} {'FocalRec':>10} {'LineCov':>10}"
    )
    print(f"  {'-' * 19} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10}")

    for category in CATEGORY_ORDER:
        if category not in quality_data:
            continue
        data = quality_data[category]
        display_name = CATEGORY_DISPLAY_NAMES.get(category, category)
        focal = data.get("focal_recall", {})
        line = data.get("line_coverage", {})
        focal_mean = focal.get("mean", 0) if focal else 0
        line_mean = line.get("mean", 0) if line else 0
        print(
            f"  {display_name:<19} {data['count']:>8} {data['compile_rate'] * 100:>9.1f}% "
            f"{focal_mean:>9.3f} {line_mean:>9.3f}"
        )


def evaluate_directory(
    output_dir: Path,
    eval_file_name: str,
    valid_ids: set[int] | None,
    bucketed_datasets: Dict[str, NL2TestDataset | None],
    bucket_test_maps: Dict[str, BucketTestMap],
) -> Dict[str, Any]:
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
    is_nl2test = is_nl2test_dir(output_dir.name)

    category_totals = compute_category_totals(entries)
    agent_categories = compute_agent_categories(entries) if is_nl2test else {}
    tool_details = compute_tool_details(entries)
    ratios = compute_ratios(category_totals)
    quality_by_category = compute_quality_by_dominant_category(entries)

    print_main_header(f"Tool Category Analysis: {output_dir.name}")
    print(f"Total entries: {len(entries)}")
    print(f"Compile rate: {compile_rate:.1%}")

    print_category_distribution(category_totals)

    if is_nl2test and agent_categories:
        print_agent_category_distribution(agent_categories)

    print_ratios(ratios)
    print_tool_details(tool_details)
    print_quality_by_dominant_category(quality_by_category)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dir_name = output_dir.name.replace("output", "").strip("_")

    result = {
        "system_name": output_dir.name,
        "total_entries": len(entries),
        "compile_rate": compile_rate,
        "category_totals": category_totals,
        "agent_categories": agent_categories if is_nl2test else None,
        "ratios": ratios,
        "tool_details": tool_details,
        "quality_by_dominant_category": quality_by_category,
    }

    output_file = OUTPUT_DIR / f"{dir_name}_category_analysis.json"
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nJSON results written to: {output_file}")

    return result


def build_cross_system_comparison(
    all_analyses: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    comparison: Dict[str, Any] = {}

    for dir_name, analysis in all_analyses.items():
        category_totals = analysis.get("category_totals", {})
        total_calls = sum(cat.get("count", 0) for cat in category_totals.values())
        entry_count = analysis.get("total_entries", 1)

        comparison[dir_name] = {
            "total_entries": entry_count,
            "compile_rate": analysis.get("compile_rate", 0),
            "total_tool_calls": total_calls,
            "calls_per_entry": total_calls / entry_count if entry_count > 0 else 0,
            "category_proportions": {
                cat: data.get("proportion", 0) for cat, data in category_totals.items()
            },
            "ratios": analysis.get("ratios", {}),
        }

    return comparison


def print_cross_system_summary(comparison: Dict[str, Any]) -> None:
    print_main_header("Cross-System Category Comparison")

    print(f"  {'System':<35} {'Entries':>8} {'Compile%':>10} {'Calls/Entry':>12}")
    print(f"  {'-' * 35} {'-' * 8} {'-' * 10} {'-' * 12}")

    for system, data in comparison.items():
        print(
            f"  {system:<35} {data.get('total_entries', 0):>8} "
            f"{data.get('compile_rate', 0) * 100:>9.1f}% "
            f"{data.get('calls_per_entry', 0):>11.1f}"
        )

    print_section_header("Category Proportions by System")

    cats = [
        "retrieval",
        "inspection",
        "generation",
        "validation",
        "orchestration",
    ]
    headers = ["Retr.", "Insp.", "Gen.", "Valid.", "Orch."]

    print(f"  {'System':<35}", end="")
    for header in headers:
        print(f" {header:>8}", end="")
    print()
    print(f"  {'-' * 35}", end="")
    for _ in headers:
        print(f" {'-' * 8}", end="")
    print()

    for system, data in comparison.items():
        proportions = data.get("category_proportions", {})
        print(f"  {system:<35}", end="")
        for cat in cats:
            pct = proportions.get(cat, 0) * 100
            print(f" {pct:>7.1f}%", end="")
        print()


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
        print_cross_system_summary(comparison)

        comparison_file = OUTPUT_DIR / "category_comparison.json"
        with comparison_file.open("w", encoding="utf-8") as handle:
            json.dump(comparison, handle, indent=2)
        print(f"\nComparison results written to: {comparison_file}")


if __name__ == "__main__":
    main()
