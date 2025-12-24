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

print("HTTP_PROXY:", os.environ.get("HTTP_PROXY"))
print("HTTPS_PROXY:", os.environ.get("HTTPS_PROXY"))


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

    # Track per-project and overall counts based on the actual filtered datasets
    # written to filtered_bucketed_tests/. This mirrors filtered_tests/summary.json
    # but reflects any tests that were dropped during bucket matching.
    project_summaries: Dict[str, Dict[str, int]] = {}
    total_test_classes = 0
    total_test_methods = 0

    def count_tests_in_dataset(dataset: NL2TestDataset) -> tuple[int, int]:
        """
        Count unique test classes and methods in a bucketed dataset.

        A "test method" is a unique (qualified_class_name, method_signature) pair.
        A "test class" is a unique qualified_class_name across all buckets.
        """
        method_pairs: set[tuple[str, str]] = set()
        buckets = [
            dataset.tests_with_one_focal_methods,
            dataset.tests_with_two_focal_methods,
            dataset.tests_with_more_than_two_to_five_focal_methods,
            dataset.tests_with_more_than_five_to_ten_focal_methods,
            dataset.tests_with_more_than_ten_focal_methods,
        ]
        for bucket in buckets:
            for test in bucket:
                method_pairs.add((test.qualified_class_name, test.method_signature))

        class_names = {class_name for class_name, _ in method_pairs}
        return (len(class_names), len(method_pairs))

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
        test_classes_and_methods = filtered_data.get("test_classes_and_methods", {})

        filtered_dataset = filter_tests_by_date(dataset, test_classes_and_methods)

        project_output_dir = output_dir / project_name
        project_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = project_output_dir / OUTPUT_FILE_NAME
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(filtered_dataset.model_dump(), f, indent=4)

        # Update summaries after writing the project dataset.
        project_test_class_count, project_test_method_count = count_tests_in_dataset(
            filtered_dataset
        )
        project_summaries[project_name] = {
            "test_class_count": project_test_class_count,
            "test_method_count": project_test_method_count,
        }
        total_test_classes += project_test_class_count
        total_test_methods += project_test_method_count

        print(f"Processed {project_name}: saved to {output_file}")

    summary_file = output_dir / SUMMARY_FILE_NAME
    summary_data = {
        "summary": {
            "total_test_classes": total_test_classes,
            "total_test_methods": total_test_methods,
        },
        "projects": project_summaries,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    print(f"\nTop-level summary saved to: {summary_file}")
    print(f"  - Total projects: {len(project_summaries)}")


def main():
    resources_dir = ROOT_DIR / RESOURCES_DIR
    bucket_dir = resources_dir / BUCKETED_TESTS_DIR
    filtered_dir = resources_dir / FILTERED_TESTS_DIR
    output_dir = resources_dir / FILTERED_BUCKETED_TESTS_DIR

    process_projects(bucket_dir, filtered_dir, output_dir)


if __name__ == "__main__":
    main()
