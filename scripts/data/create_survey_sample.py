"""Script to create survey sample from filtered bucketed dataset and evaluation results."""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cldk import CLDK
from cldk.analysis import AnalysisLevel
from cldk.analysis.java import JavaAnalysis

from sakura.utils.analysis.java_analyzer import CommonAnalysis


@dataclass
class EntryDisplayInfo:
    """Display info for a survey entry (not saved to JSON)."""

    entry_id: int
    bucket: str
    qualified_class_name: str
    method_signature: str
    gt_ncloc: int
    generated_ncloc: int
    structured_eval: dict | None
    coverage_eval: dict | None

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_STATS_DIR = ROOT_DIR / "resources" / "filtered_bucketed_dataset_stats"
BUCKETED_DIR = ROOT_DIR / "resources" / "filtered_bucketed_tests"
NL2TEST_OUTPUT_DIR = ROOT_DIR / "resources" / "outputs" / "nl2test_gemini_flash_output"
ANALYSIS_DIR = ROOT_DIR / "resources" / "analysis"
DATASETS_DIR = ROOT_DIR / "resources" / "datasets"
OUTPUT_DIR = ROOT_DIR / "resources" / "survey_sample"

MAX_LINES = 50

BUCKET_NAMES = [
    "tests_with_one_focal_methods",
    "tests_with_two_focal_methods",
    "tests_with_more_than_two_to_five_focal_methods",
    "tests_with_more_than_five_to_ten_focal_methods",
    "tests_with_more_than_ten_focal_methods",
]


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def load_java_analysis(project_name: str) -> JavaAnalysis | None:
    """Load JavaAnalysis for a project if analysis exists."""
    analysis_path = ANALYSIS_DIR / project_name
    analysis_json = analysis_path / "analysis.json"
    project_path = DATASETS_DIR / project_name

    if not analysis_json.exists() or not project_path.exists():
        return None

    try:
        return CLDK(language="java").analysis(
            project_path=str(project_path),
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=str(analysis_path),
            eager=False,
        )
    except Exception:
        return None


def get_projects_sorted_by_focal_methods(stats_dir: Path) -> list[dict]:
    """Load summary.json and return projects sorted by average focal methods (descending)."""
    summary = load_json(stats_dir / "summary.json")
    projects = summary["projects"]
    return sorted(
        projects,
        key=lambda p: p["focal_methods_distribution"]["mean"],
        reverse=True,
    )


def get_non_empty_buckets(dataset: dict) -> list[str]:
    """Return bucket names that have at least one test, ordered from highest to lowest."""
    non_empty = []
    for bucket_name in reversed(BUCKET_NAMES):
        if dataset.get(bucket_name) and len(dataset[bucket_name]) > 0:
            non_empty.append(bucket_name)
    return non_empty


def print_project_header(project_name: str, avg_focal: float) -> None:
    """Print a project header box."""
    title = f" {project_name} (avg focal methods: {avg_focal:.2f}) "
    width = max(len(title) + 4, 60)
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_entry_box(info: EntryDisplayInfo) -> None:
    """Print a formatted box for a survey entry."""
    width = 70
    print("+" + "-" * (width - 2) + "+")
    print(f"| Entry #{info.entry_id:<{width - 12}}|")
    print("+" + "-" * (width - 2) + "+")

    # Bucket
    bucket_short = info.bucket.replace("tests_with_", "").replace("_focal_methods", " FM")
    print(f"| Bucket: {bucket_short:<{width - 12}}|")

    # Class and method
    class_display = info.qualified_class_name
    if len(class_display) > width - 12:
        class_display = "..." + class_display[-(width - 15) :]
    print(f"| Class:  {class_display:<{width - 12}}|")

    method_display = info.method_signature
    if len(method_display) > width - 12:
        method_display = method_display[: width - 15] + "..."
    print(f"| Method: {method_display:<{width - 12}}|")

    # NCLOC
    ncloc_line = f"GT NCLOC: {info.gt_ncloc}    Generated NCLOC: {info.generated_ncloc}"
    print(f"| {ncloc_line:<{width - 4}}|")

    print("+" + "-" * (width - 2) + "+")

    # Structured eval
    if info.structured_eval:
        print(f"| {'STRUCTURED EVAL':<{width - 4}}|")
        se = info.structured_eval
        print(
            f"|   Focal:      P={se.get('focal_precision', 0):.2f}  "
            f"R={se.get('focal_recall', 0):.2f}"
            + " " * (width - 38)
            + "|"
        )
        print(
            f"|   Callable:   P={se.get('callable_precision', 0):.2f}  "
            f"R={se.get('callable_recall', 0):.2f}"
            + " " * (width - 38)
            + "|"
        )
        print(
            f"|   Assertion:  P={se.get('assertion_precision', 0):.2f}  "
            f"R={se.get('assertion_recall', 0):.2f}"
            + " " * (width - 38)
            + "|"
        )
        print(
            f"|   ObjCreate:  P={se.get('obj_creation_precision', 0):.2f}  "
            f"R={se.get('obj_creation_recall', 0):.2f}"
            + " " * (width - 38)
            + "|"
        )
    else:
        print(f"| {'STRUCTURED EVAL: N/A':<{width - 4}}|")

    print("+" + "-" * (width - 2) + "+")

    # Coverage eval
    if info.coverage_eval:
        print(f"| {'COVERAGE EVAL':<{width - 4}}|")
        ce = info.coverage_eval
        print(
            f"|   Class: {ce.get('class_coverage', 0):>6.1%}   "
            f"Method: {ce.get('method_coverage', 0):>6.1%}   "
            f"Line: {ce.get('line_coverage', 0):>6.1%}   "
            f"Branch: {ce.get('branch_coverage', 0):>6.1%} |"
        )
    else:
        print(f"| {'COVERAGE EVAL: N/A':<{width - 4}}|")

    print("+" + "-" * (width - 2) + "+")


def extract_test_method_code(code: str) -> str | None:
    """
    Extract the test method code from a full test class.
    Returns the method including annotation, signature, and body.
    Returns None if no test method found or extraction fails.
    """
    lines = code.split("\n")

    # Find line with test annotation
    test_start_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith("@Test")
            or stripped.startswith("@ParameterizedTest")
            or stripped.startswith("@RepeatedTest")
        ):
            test_start_idx = i
            break

    if test_start_idx is None:
        return None

    # Find the method body by counting braces
    method_lines = []
    brace_depth = 0
    method_started = False

    for i in range(test_start_idx, len(lines)):
        line = lines[i]
        method_lines.append(line)

        for char in line:
            if char == "{":
                brace_depth += 1
                method_started = True
            elif char == "}":
                brace_depth -= 1

        if method_started and brace_depth == 0:
            break

    if not method_started or brace_depth != 0:
        return None

    return "\n".join(method_lines)


def find_compiling_eval(
    tests: list[dict],
    eval_results: list[dict],
    analysis: JavaAnalysis,
    common_analysis: CommonAnalysis,
) -> tuple[dict, dict, int, int] | None:
    """
    Find a compiling evaluation result for a test in the given list.
    Returns (test, eval_result, gt_ncloc, generated_ncloc) tuple or None if not found.

    Criteria:
    - compiles must be True
    - Generated test method must have NCLOC <= MAX_LINES
    - Ground truth test method must have NCLOC <= MAX_LINES
    """
    random.shuffle(tests)

    eval_lookup: dict[tuple[str, str], dict] = {}
    for eval_result in eval_results:
        nl2test_input = eval_result.get("nl2test_input", {})
        if nl2test_input.get("abstraction_level") == "medium":
            key = (
                nl2test_input.get("qualified_class_name", ""),
                nl2test_input.get("method_signature", ""),
            )
            eval_lookup[key] = eval_result

    for test in tests:
        key = (test["qualified_class_name"], test["method_signature"])
        eval_result = eval_lookup.get(key)

        if not eval_result or eval_result.get("compiles") is not True:
            continue

        # Check generated test method NCLOC
        nl2test_metadata = eval_result.get("nl2test_metadata", {})
        generated_code = nl2test_metadata.get("code", "")
        if not generated_code:
            continue
        test_method_code = extract_test_method_code(generated_code)
        if test_method_code is None:
            continue
        generated_ncloc = common_analysis.get_ncloc("", test_method_code)
        if generated_ncloc > MAX_LINES:
            continue

        # Check ground truth NCLOC
        qualified_class_name = test["qualified_class_name"]
        method_signature = test["method_signature"]
        method_details = analysis.get_method(qualified_class_name, method_signature)
        if not method_details:
            continue
        gt_ncloc = common_analysis.get_ncloc(
            method_details.declaration, method_details.code
        )
        if gt_ncloc > MAX_LINES:
            continue

        return test, eval_result, gt_ncloc, generated_ncloc

    return None


def process_project(
    project_name: str,
    bucketed_dir: Path,
    nl2test_output_dir: Path,
) -> tuple[list[dict], list[EntryDisplayInfo]] | None:
    """
    Process a single project and return survey entries and display info.
    Returns None if criteria not met.
    """
    nl2test_path = bucketed_dir / project_name / "nl2test.json"
    eval_path = nl2test_output_dir / project_name / "nl2test_evaluation_results.json"

    if not nl2test_path.exists() or not eval_path.exists():
        return None

    dataset = load_json(nl2test_path)
    eval_results = load_json(eval_path)

    non_empty_buckets = get_non_empty_buckets(dataset)
    if len(non_empty_buckets) < 3:
        return None

    # Load analysis for NCLOC checks
    analysis = load_java_analysis(project_name)
    if analysis is None:
        return None

    common_analysis = CommonAnalysis(analysis)

    entries = []
    display_infos = []
    used_buckets = set()
    entry_id = 1

    for bucket_name in non_empty_buckets:
        if len(used_buckets) >= 3:
            break

        tests = list(dataset[bucket_name])
        result = find_compiling_eval(tests, eval_results, analysis, common_analysis)

        if result:
            test, eval_result, gt_ncloc, generated_ncloc = result
            nl2test_input = eval_result["nl2test_input"]

            entry = {
                "bucket": bucket_name,
                "nl2test_input": nl2test_input,
                "nl2test_output": eval_result["nl2test_metadata"],
            }
            entries.append(entry)

            display_info = EntryDisplayInfo(
                entry_id=entry_id,
                bucket=bucket_name,
                qualified_class_name=nl2test_input.get("qualified_class_name", ""),
                method_signature=nl2test_input.get("method_signature", ""),
                gt_ncloc=gt_ncloc,
                generated_ncloc=generated_ncloc,
                structured_eval=eval_result.get("structured_eval"),
                coverage_eval=eval_result.get("coverage_eval"),
            )
            display_infos.append(display_info)

            used_buckets.add(bucket_name)
            entry_id += 1

    if len(entries) < 3:
        return None

    return entries, display_infos


def main():
    random.seed(42)

    sorted_projects = get_projects_sorted_by_focal_methods(DATASET_STATS_DIR)

    survey_sample: dict[str, list[dict]] = {}

    for project_info in sorted_projects:
        project_name = project_info["project_name"]
        avg_focal = project_info["focal_methods_distribution"]["mean"]

        result = process_project(project_name, BUCKETED_DIR, NL2TEST_OUTPUT_DIR)

        if result:
            entries, display_infos = result
            survey_sample[project_name] = entries

            print_project_header(project_name, avg_focal)
            for info in display_infos:
                print_entry_box(info)

            if len(survey_sample) >= 3:
                break
        else:
            print(f"Skipped {project_name} (avg focal methods: {avg_focal:.2f})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "survey_sample.json"
    with open(output_path, "w") as f:
        json.dump(survey_sample, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Saved {len(survey_sample)} projects to {output_path}")


if __name__ == "__main__":
    main()
