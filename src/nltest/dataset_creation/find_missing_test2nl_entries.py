import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import ray
from tqdm import tqdm

from nltest.dataset_creation.model import NL2TestDataset, Test

# Path constants
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # Project root

# Test2NL dataset paths
TEST2NL_DIR = "resources/test2nl/filtered_dataset"
TEST2NL_FILE = "test2nl.csv"

# Bucketed dataset paths
BUCKETED_DATASET_DIR = "resources/filtered_bucketed_tests"
BUCKETED_FILE = "nl2test.json"

# Output paths
MISSING_TESTS_DIR = "resources/missing_tests"
OUTPUT_FILE = "nl2test.json"


def load_test2nl_entries(csv_path: Path) -> Set[Tuple[str, str, str]]:
    """
    Load Test2NL entries from CSV and return a set of
    (project_name, qualified_class_name, method_signature) tuples.
    """
    entries: Set[Tuple[str, str, str]] = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.add(
                (
                    row["project_name"],
                    row["qualified_class_name"],
                    row["method_signature"],
                )
            )
    return entries


def get_all_tests_by_bucket(dataset: NL2TestDataset) -> Dict[str, List[Test]]:
    """Return tests organized by bucket name."""
    return {
        "tests_with_one_focal_methods": dataset.tests_with_one_focal_methods,
        "tests_with_two_focal_methods": dataset.tests_with_two_focal_methods,
        "tests_with_more_than_two_to_five_focal_methods": (
            dataset.tests_with_more_than_two_to_five_focal_methods
        ),
        "tests_with_more_than_five_to_ten_focal_methods": (
            dataset.tests_with_more_than_five_to_ten_focal_methods
        ),
        "tests_with_more_than_ten_focal_methods": (
            dataset.tests_with_more_than_ten_focal_methods
        ),
    }


@ray.remote
def find_missing_tests_for_project(
    project_name: str,
    bucketed_file: str,
    test2nl_entries: Set[Tuple[str, str, str]],
    output_dir: str,
) -> Dict[str, Any] | None:
    """
    Find tests in the bucketed dataset that are missing from Test2NL entries.

    Returns a dict with project stats or None if processing fails.
    """
    with open(bucketed_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = NL2TestDataset(**data)

    tests_by_bucket = get_all_tests_by_bucket(dataset)
    missing_by_bucket: Dict[str, List[Test]] = {k: [] for k in tests_by_bucket}

    total_tests = 0
    missing_count = 0

    for bucket_name, tests in tests_by_bucket.items():
        for test in tests:
            total_tests += 1
            key = (project_name, test.qualified_class_name, test.method_signature)
            if key not in test2nl_entries:
                missing_by_bucket[bucket_name].append(test)
                missing_count += 1

    if missing_count == 0:
        return {
            "project_name": project_name,
            "total_tests": total_tests,
            "missing_count": 0,
        }

    # Create output dataset with missing tests
    missing_dataset = NL2TestDataset(
        dataset_name=f"{project_name}_missing",
        tests_with_one_focal_methods=missing_by_bucket["tests_with_one_focal_methods"],
        tests_with_two_focal_methods=missing_by_bucket["tests_with_two_focal_methods"],
        tests_with_more_than_two_to_five_focal_methods=missing_by_bucket[
            "tests_with_more_than_two_to_five_focal_methods"
        ],
        tests_with_more_than_five_to_ten_focal_methods=missing_by_bucket[
            "tests_with_more_than_five_to_ten_focal_methods"
        ],
        tests_with_more_than_ten_focal_methods=missing_by_bucket[
            "tests_with_more_than_ten_focal_methods"
        ],
    )

    # Write output
    project_output_dir = Path(output_dir) / project_name
    project_output_dir.mkdir(parents=True, exist_ok=True)
    output_file = project_output_dir / OUTPUT_FILE

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(missing_dataset.model_dump(), f, indent=2)

    return {
        "project_name": project_name,
        "total_tests": total_tests,
        "missing_count": missing_count,
    }


def create_summary(output_dir: Path, results: List[Dict[str, Any] | None]) -> None:
    """Create a summary.json with missing test stats per project."""
    summary: Dict[str, Any] = {
        "total_projects": 0,
        "projects_with_missing": 0,
        "total_missing_tests": 0,
        "total_tests": 0,
        "projects": {},
    }

    for result in results:
        if result is None:
            continue

        summary["total_projects"] += 1
        summary["total_tests"] += result["total_tests"]
        missing_count = result["missing_count"]

        if missing_count > 0:
            summary["projects_with_missing"] += 1
            summary["total_missing_tests"] += missing_count
            summary["projects"][result["project_name"]] = {
                "total": result["total_tests"],
                "missing": missing_count,
            }

    summary["projects"] = dict(sorted(summary["projects"].items()))

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nMissing tests analysis complete:")
    print(f"  Total projects: {summary['total_projects']}")
    print(f"  Projects with missing tests: {summary['projects_with_missing']}")
    print(f"  Total tests: {summary['total_tests']}")
    print(f"  Total missing tests: {summary['total_missing_tests']}")
    print(f"  Summary saved to: {summary_file}")


def process_projects(
    test2nl_file: Path,
    bucketed_dir: Path,
    output_dir: Path,
) -> None:
    """Process all projects in parallel to find missing tests."""
    # Clear previous output
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Test2NL entries
    print(f"Loading Test2NL entries from {test2nl_file}...")
    test2nl_entries = load_test2nl_entries(test2nl_file)
    print(f"  Loaded {len(test2nl_entries)} unique test entries")

    # Put entries in Ray object store for sharing across workers
    test2nl_ref = ray.put(test2nl_entries)

    # Collect projects and launch ray tasks
    futures = []
    for project_name in sorted(os.listdir(bucketed_dir)):
        if project_name.startswith(".") or project_name.startswith("__"):
            continue

        project_path = bucketed_dir / project_name
        if not project_path.is_dir():
            continue

        bucketed_file = project_path / BUCKETED_FILE
        if not bucketed_file.exists():
            print(f"Skipping {project_name}: no {BUCKETED_FILE} found")
            continue

        futures.append(
            find_missing_tests_for_project.remote(  # pyright: ignore[reportAttributeAccessIssue]
                project_name,
                str(bucketed_file),
                test2nl_ref,
                str(output_dir),
            )
        )

    # Process results as they complete
    results: List[Dict[str, Any] | None] = []
    with tqdm(total=len(futures), desc="Analyzing projects...") as pbar:
        while futures:
            done, futures = ray.wait(futures, num_returns=1)
            res = ray.get(done)
            results.extend(res)
            pbar.update(len(done))

    create_summary(output_dir, results)


def main():
    test2nl_file = ROOT_DIR / TEST2NL_DIR / TEST2NL_FILE
    bucketed_dir = ROOT_DIR / BUCKETED_DATASET_DIR
    output_dir = ROOT_DIR / MISSING_TESTS_DIR

    if not test2nl_file.exists():
        raise FileNotFoundError(f"Test2NL file not found: {test2nl_file}")

    if not bucketed_dir.exists():
        raise FileNotFoundError(f"Bucketed dataset directory not found: {bucketed_dir}")

    process_projects(test2nl_file, bucketed_dir, output_dir)


if __name__ == "__main__":
    main()
