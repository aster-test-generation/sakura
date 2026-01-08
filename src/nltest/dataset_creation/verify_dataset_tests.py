import json
import os
from pathlib import Path
from typing import Any, Dict, List

from cldk import CLDK
from cldk.analysis import AnalysisLevel

from nltest.dataset_creation.model import NL2TestDataset, Test

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


def verify_project_tests(
    dataset: NL2TestDataset,
    analysis_path: Path,
    project_path: Path,
) -> List[Dict[str, str]]:
    """
    Verify all tests in a dataset can be found via CLDK analysis.

    Returns a list of erroneous tests (those where get_method returns falsy).
    """
    try:
        analysis = CLDK(
            language="java"
        ).analysis(
            project_path=str(project_path),
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=str(analysis_path),
            eager=True,  # Reloads to ensure it is compliant with most recent project status
        )
    except Exception as e:
        print(f"  Failed to load analysis: {e}")
        return []

    erroneous: List[Dict[str, str]] = []
    all_tests = get_all_tests(dataset)

    for test in all_tests:
        method = analysis.get_method(test.qualified_class_name, test.method_signature)
        if not method:
            erroneous.append(
                {
                    "qualified_class_name": test.qualified_class_name,
                    "method_signature": test.method_signature,
                }
            )

    return erroneous


def process_projects(
    tests_dir: Path,
    analysis_dir: Path,
    datasets_dir: Path,
    output_dir: Path,
) -> None:
    """
    Process all projects in tests_dir, verify tests against analysis, and output erroneous tests.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "total_projects": 0,
        "projects_with_errors": 0,
        "total_erroneous_tests": 0,
        "projects": {},
    }

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

        print(f"Processing {project_name}...")

        with open(tests_file, "r", encoding="utf-8") as f:
            tests_data = json.load(f)
        dataset = NL2TestDataset(**tests_data)

        erroneous_tests = verify_project_tests(
            dataset,
            project_analysis_path,
            project_source_path,
        )

        summary["total_projects"] += 1

        if erroneous_tests:
            summary["projects_with_errors"] += 1
            summary["total_erroneous_tests"] += len(erroneous_tests)
            summary["projects"][project_name] = len(erroneous_tests)

            # Write erroneous tests for this project
            project_output_dir = output_dir / project_name
            project_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = project_output_dir / OUTPUT_FILE_NAME

            erroneous_data = {
                "project_name": project_name,
                "erroneous_test_count": len(erroneous_tests),
                "erroneous_tests": erroneous_tests,
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(erroneous_data, f, indent=2)

            print(f"  Found {len(erroneous_tests)} erroneous tests -> {output_file}")
        else:
            print("  All tests verified successfully")

    # Write summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nVerification complete:")
    print(f"  Total projects: {summary['total_projects']}")
    print(f"  Projects with errors: {summary['projects_with_errors']}")
    print(f"  Total erroneous tests: {summary['total_erroneous_tests']}")
    print(f"  Summary saved to: {summary_file}")


def main():
    resources_dir = ROOT_DIR / RESOURCES_DIR
    tests_dir = resources_dir / TESTS_DIR
    analysis_dir = resources_dir / ANALYSIS_DIR
    datasets_dir = resources_dir / DATASETS_DIR
    output_dir = resources_dir / ERRONEOUS_TESTS_DIR

    process_projects(tests_dir, analysis_dir, datasets_dir, output_dir)


if __name__ == "__main__":
    main()
