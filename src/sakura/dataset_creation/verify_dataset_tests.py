import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from tqdm import tqdm

from sakura.dataset_creation.model import NL2TestDataset, Test
from sakura.utils.analysis.java_analyzer import CommonAnalysis

# Path constants
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # Project root
INPUT_FILE_NAME = "nl2test.json"
OUTPUT_FILE_NAME = "erroneous_tests.json"

# Resources paths (relative to ROOT_DIR)
RESOURCES_DIR = "resources"
TESTS_DIR = "filtered_bucketed_tests"  # Input: contains nl2test.json per project
ANALYSIS_DIR = "analysis"  # Contains analysis.json per project
DATASETS_DIR = "datasets"  # Source projects for CLDK analysis
ERRONEOUS_TESTS_DIR = "erroneous_tests"  # Output: erroneous tests per project


def get_all_tests(dataset: NL2TestDataset) -> List[Test]:
    """Collect all tests from all buckets in the dataset."""
    return (
        dataset.tests_with_one_focal_methods
        + dataset.tests_with_two_focal_methods
        + dataset.tests_with_more_than_two_to_five_focal_methods
        + dataset.tests_with_more_than_five_to_ten_focal_methods
        + dataset.tests_with_more_than_ten_focal_methods
    )


@ray.remote
def _verify_project(
    project_name: str,
    tests_file: str,
    analysis_path: str,
    project_path: str,
    output_dir: str,
) -> Dict[str, Any] | None:
    """
    Verify all tests in a project can be found via CLDK analysis.

    Returns verification result dict or None if processing fails.
    """
    with open(tests_file, "r", encoding="utf-8") as f:
        tests_data = json.load(f)
    dataset = NL2TestDataset(**tests_data)

    try:
        analysis = CLDK(
            language="java"
        ).analysis(
            project_path=project_path,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=analysis_path,
            eager=True,  # Reloads to ensure it is compliant with most recent project status
        )
    except Exception:
        return None

    erroneous: List[Dict[str, str]] = []
    all_tests = get_all_tests(dataset)

    for test in all_tests:
        method = analysis.get_method(test.qualified_class_name, test.method_signature)
        if not method:
            # Try simplified signature for cases with fully qualified types
            simplified_sig = CommonAnalysis.simplify_method_signature(
                test.method_signature
            )
            method = analysis.get_method(test.qualified_class_name, simplified_sig)

        if not method:
            erroneous.append(
                {
                    "qualified_class_name": test.qualified_class_name,
                    "method_signature": test.method_signature,
                }
            )

    # Write erroneous tests if any found
    if erroneous:
        project_output_dir = Path(output_dir) / project_name
        project_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = project_output_dir / OUTPUT_FILE_NAME

        erroneous_data = {
            "project_name": project_name,
            "erroneous_test_count": len(erroneous),
            "erroneous_tests": erroneous,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(erroneous_data, f, indent=2)

    return {
        "project_name": project_name,
        "erroneous_count": len(erroneous),
    }


def _create_summary(output_dir: Path, results: List[Dict[str, Any] | None]) -> None:
    """Create a summary.json with verification results per project."""
    summary: Dict[str, Any] = {
        "total_projects": 0,
        "projects_with_errors": 0,
        "total_erroneous_tests": 0,
        "projects": {},
    }

    for result in results:
        if result is None:
            continue

        summary["total_projects"] += 1
        erroneous_count = result["erroneous_count"]

        if erroneous_count > 0:
            summary["projects_with_errors"] += 1
            summary["total_erroneous_tests"] += erroneous_count
            summary["projects"][result["project_name"]] = erroneous_count

    summary["projects"] = dict(sorted(summary["projects"].items()))

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nVerification complete:")
    print(f"  Total projects: {summary['total_projects']}")
    print(f"  Projects with errors: {summary['projects_with_errors']}")
    print(f"  Total erroneous tests: {summary['total_erroneous_tests']}")
    print(f"  Summary saved to: {summary_file}")


def process_projects(
    tests_dir: Path,
    analysis_dir: Path,
    datasets_dir: Path,
    output_dir: Path,
) -> None:
    """
    Process all projects in tests_dir in parallel, verify tests against analysis,
    and output erroneous tests.
    """
    # Clear previous results
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect valid projects and launch ray tasks
    futures = []
    for project_name in sorted(os.listdir(tests_dir)):
        if project_name.startswith(".") or project_name.startswith("__"):
            continue

        project_tests_path = tests_dir / project_name
        if not project_tests_path.is_dir():
            continue

        tests_file = project_tests_path / INPUT_FILE_NAME
        if not tests_file.exists():
            print(f"Skipping {project_name}: no {INPUT_FILE_NAME} in tests directory")
            continue

        project_analysis_path = analysis_dir / project_name
        if not project_analysis_path.exists():
            print(f"Skipping {project_name}: no analysis directory found")
            continue

        project_source_path = datasets_dir / project_name
        if not project_source_path.exists():
            print(f"Skipping {project_name}: no source project found in datasets")
            continue

        futures.append(
            _verify_project.remote(  # pyright: ignore[reportAttributeAccessIssue]
                project_name,
                str(tests_file),
                str(project_analysis_path),
                str(project_source_path),
                str(output_dir),
            )
        )

    # Process results as they complete
    results: List[Dict[str, Any] | None] = []
    with tqdm(total=len(futures), desc="Verifying projects...") as pbar:
        while futures:
            done, futures = ray.wait(futures, num_returns=1)
            res = ray.get(done)
            results.extend(res)
            pbar.update(len(done))

    _create_summary(output_dir, results)


def main():
    resources_dir = ROOT_DIR / RESOURCES_DIR
    tests_dir = resources_dir / TESTS_DIR
    analysis_dir = resources_dir / ANALYSIS_DIR
    datasets_dir = resources_dir / DATASETS_DIR
    output_dir = resources_dir / ERRONEOUS_TESTS_DIR

    process_projects(tests_dir, analysis_dir, datasets_dir, output_dir)


if __name__ == "__main__":
    main()
