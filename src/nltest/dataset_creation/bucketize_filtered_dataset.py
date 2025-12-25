import json
import os
from pathlib import Path
from typing import Dict, List

from nltest.dataset_creation.model import NL2TestDataset, Test

# Path constants
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # Project root
INPUT_FILE_NAME = "nl2test.json"
OUTPUT_FILE_NAME = "nl2test.json"
SUMMARY_FILE_NAME = "summary.json"

# Resources paths
RESOURCES_DIR = "resources"  # Relative to ROOT_DIR
BUCKETED_TESTS_DIR = "bucketed_tests"  # Relative to RESOURCES_DIR
FILTERED_TESTS_DIR = "filtered_tests"  # Relative to RESOURCES_DIR
FILTERED_BUCKETED_TESTS_DIR = "filtered_bucketed_tests"  # Relative to RESOURCES_DIR


def filter_tests_by_date(
    dataset: NL2TestDataset,
    test_classes_and_methods: Dict[str, List[str]],
) -> NL2TestDataset:
    """
    Filter tests from an NL2TestDataset to only include those present in filtered tests.

    Args:
        dataset: The original NL2TestDataset to filter.
        test_classes_and_methods: Dict mapping qualified class names to method signatures.

    Returns:
        New NL2TestDataset containing only tests present in test_classes_and_methods.
    """

    def filter_bucket(tests: List[Test]) -> List[Test]:
        filtered = []
        for test in tests:
            if test.qualified_class_name in test_classes_and_methods:
                methods = test_classes_and_methods[test.qualified_class_name]
                if test.method_signature in methods:
                    filtered.append(test)
        return filtered

    return NL2TestDataset(
        dataset_name=dataset.dataset_name,
        tests_with_one_focal_methods=filter_bucket(
            dataset.tests_with_one_focal_methods
        ),
        tests_with_two_focal_methods=filter_bucket(
            dataset.tests_with_two_focal_methods
        ),
        tests_with_more_than_two_to_five_focal_methods=filter_bucket(
            dataset.tests_with_more_than_two_to_five_focal_methods
        ),
        tests_with_more_than_five_to_ten_focal_methods=filter_bucket(
            dataset.tests_with_more_than_five_to_ten_focal_methods
        ),
        tests_with_more_than_ten_focal_methods=filter_bucket(
            dataset.tests_with_more_than_ten_focal_methods
        ),
    )


def process_projects(bucket_dir: Path, filtered_dir: Path, output_dir: Path) -> None:
    """
    Process all projects in bucket_dir and filter them against filtered_dir.

    Args:
        bucket_dir: Directory containing bucketed NL2TestDataset files per project.
        filtered_dir: Directory containing filtered test classes and methods per project.
        output_dir: Directory to write filtered bucketed datasets.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    bucket_names = [
        "tests_with_one_focal_methods",
        "tests_with_two_focal_methods",
        "tests_with_more_than_two_to_five_focal_methods",
        "tests_with_more_than_five_to_ten_focal_methods",
        "tests_with_more_than_ten_focal_methods",
    ]
    totals: Dict[str, int] = {bucket: 0 for bucket in bucket_names}
    totals["total"] = 0
    projects: Dict[str, Dict[str, int]] = {}

    def get_bucket_counts(dataset: NL2TestDataset) -> Dict[str, int]:
        """Get per-bucket test counts for a dataset."""
        counts = {
            "tests_with_one_focal_methods": len(dataset.tests_with_one_focal_methods),
            "tests_with_two_focal_methods": len(dataset.tests_with_two_focal_methods),
            "tests_with_more_than_two_to_five_focal_methods": len(
                dataset.tests_with_more_than_two_to_five_focal_methods
            ),
            "tests_with_more_than_five_to_ten_focal_methods": len(
                dataset.tests_with_more_than_five_to_ten_focal_methods
            ),
            "tests_with_more_than_ten_focal_methods": len(
                dataset.tests_with_more_than_ten_focal_methods
            ),
        }
        counts["total"] = sum(counts.values())
        return counts

    for project_name in os.listdir(bucket_dir):
        if project_name.startswith(".") or project_name.startswith("__"):
            continue

        project_bucket_path = bucket_dir / project_name
        if not project_bucket_path.is_dir():
            continue

        project_filtered_path = filtered_dir / project_name
        if not project_filtered_path.exists():
            print(f"Skipping {project_name}: not found in filtered_tests/")
            continue

        bucket_file = project_bucket_path / INPUT_FILE_NAME
        if not bucket_file.exists():
            print(f"Skipping {project_name}: no {INPUT_FILE_NAME} in bucketed_tests/")
            continue

        with open(bucket_file, "r", encoding="utf-8") as f:
            bucket_data = json.load(f)
        dataset = NL2TestDataset(**bucket_data)

        filtered_file = project_filtered_path / INPUT_FILE_NAME
        if not filtered_file.exists():
            print(f"Skipping {project_name}: no {INPUT_FILE_NAME} in filtered_tests/")
            continue

        with open(filtered_file, "r", encoding="utf-8") as f:
            filtered_data = json.load(f)
        test_classes_and_methods = filtered_data.get("new_tests", {})

        filtered_dataset = filter_tests_by_date(dataset, test_classes_and_methods)

        project_output_dir = output_dir / project_name
        project_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = project_output_dir / OUTPUT_FILE_NAME
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(filtered_dataset.model_dump(), f, indent=4)

        # Update summaries after writing the project dataset.
        project_counts = get_bucket_counts(filtered_dataset)
        projects[project_name] = project_counts
        for bucket in bucket_names:
            totals[bucket] += project_counts[bucket]
        totals["total"] += project_counts["total"]

        print(f"Processed {project_name}: saved to {output_file}")

    summary_file = output_dir / SUMMARY_FILE_NAME
    summary_data = {
        "totals": totals,
        "projects": dict(sorted(projects.items())),
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nTop-level summary saved to: {summary_file}")
    print(f"  - Total projects: {len(projects)}")


def main():
    resources_dir = ROOT_DIR / RESOURCES_DIR
    bucket_dir = resources_dir / BUCKETED_TESTS_DIR
    filtered_dir = resources_dir / FILTERED_TESTS_DIR
    output_dir = resources_dir / FILTERED_BUCKETED_TESTS_DIR

    process_projects(bucket_dir, filtered_dir, output_dir)


if __name__ == "__main__":
    main()
