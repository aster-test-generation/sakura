import json
from pathlib import Path
from typing import Any, Dict, List

import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from cldk.analysis.java import JavaAnalysis
from hamster.code_analysis.model.models import ProjectAnalysis
from tqdm import tqdm

from nltest.dataset_creation.create_hamster_model import get_subfolders
from nltest.dataset_creation.model import NL2TestDataset, Test

# Path constants
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Relative to ROOT_DIR
RESOURCES_DIR = "resources"

# Relative to RESOURCES_DIR
DATASETS_DIR = "datasets/"  # Source projects for CLDK analysis
HAMSTER_DIR = "hamster/"  # Input: contains hamster.json per project
BUCKETED_DIR = "bucketed_tests/"  # Output: nl2test.json per project
ANALYSIS_DIR = "analysis/"  # CLDK analysis cache directory

# File names
INPUT_FILE_NAME = "hamster.json"  # Located in HAMSTER_DIR/<project>/
OUTPUT_FILE_NAME = "nl2test.json"  # Output to BUCKETED_DIR/<project>/
SUMMARY_FILE_NAME = "summary.json"  # Output to BUCKETED_DIR/


def _should_skip_test_class(
    analysis: JavaAnalysis,
    qualified_class_name: str,
) -> bool:
    """Determine if a test class should be skipped from bucketing."""
    class_details = analysis.get_class(qualified_class_name)
    if not class_details:
        return True

    # Skip abstract classes
    if class_details.modifiers and "abstract" in class_details.modifiers:
        return True

    # Skip interfaces
    if class_details.is_interface:
        return True

    # Skip annotation declarations
    if class_details.is_annotation_declaration:
        return True

    # Skip enum declarations
    if class_details.is_enum_declaration:
        return True

    return False


@ray.remote
def _create_bucketized_dataset(
    hamster_path: str,
    output_path: str,
    analysis_path: str,
    project_path: str,
) -> Dict[str, Any] | None:
    """Create a bucketized dataset from hamster analysis, filtering invalid test classes."""
    parent = Path(hamster_path).parent
    parent_name = parent.name
    if parent_name == "__pycache__" or parent_name.startswith("."):
        return

    with open(hamster_path, "r") as f:
        file_content = json.load(f)
        project_analysis = ProjectAnalysis.model_validate(file_content)

    if not project_analysis:
        return

    dataset_name = project_analysis.dataset_name
    if (
        not dataset_name
        or dataset_name == "__pycache__"
        or dataset_name.startswith(".")
    ):
        return

    try:
        analysis = CLDK(language="java").analysis(
            project_path=project_path,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=analysis_path,
            eager=False,
        )
    except Exception:
        return  # Skip project if CLDK analysis fails

    tests_with_one_focal_methods: List[Test] = []
    tests_with_two_focal_methods: List[Test] = []
    tests_with_more_than_two_to_five_focal_methods: List[Test] = []
    tests_with_more_than_five_to_ten_focal_methods: List[Test] = []
    tests_with_more_than_ten_focal_methods: List[Test] = []

    for test_class in project_analysis.test_class_analyses:
        # Skip classes that shouldn't be processed
        if _should_skip_test_class(analysis, test_class.qualified_class_name):
            continue

        for test_method in test_class.test_method_analyses:
            # TODO: Should determine if we should add a _should_skip_test_method method here, for deprecated tests or others

            focal_classes = test_method.focal_classes or []
            focal_method_count = sum(
                len(focal_class.focal_method_names or [])
                for focal_class in focal_classes
            )
            test = Test(
                qualified_class_name=test_class.qualified_class_name,
                method_signature=test_method.method_signature,
                focal_details=test_method.focal_classes,
            )
            if focal_method_count == 1:
                tests_with_one_focal_methods.append(test)
            elif focal_method_count == 2:
                tests_with_two_focal_methods.append(test)
            elif 2 < focal_method_count <= 5:
                tests_with_more_than_two_to_five_focal_methods.append(test)
            elif 5 < focal_method_count <= 10:
                tests_with_more_than_five_to_ten_focal_methods.append(test)
            elif focal_method_count > 10:
                tests_with_more_than_ten_focal_methods.append(test)

    nl2test_dataset = NL2TestDataset(
        dataset_name=dataset_name,
        tests_with_one_focal_methods=tests_with_one_focal_methods,
        tests_with_two_focal_methods=tests_with_two_focal_methods,
        tests_with_more_than_two_to_five_focal_methods=tests_with_more_than_two_to_five_focal_methods,
        tests_with_more_than_five_to_ten_focal_methods=tests_with_more_than_five_to_ten_focal_methods,
        tests_with_more_than_ten_focal_methods=tests_with_more_than_ten_focal_methods,
    )

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / OUTPUT_FILE_NAME, "w") as f:
        f.write(nl2test_dataset.model_dump_json())

    return {
        "dataset_name": dataset_name,
        "tests_with_one_focal_methods": len(tests_with_one_focal_methods),
        "tests_with_two_focal_methods": len(tests_with_two_focal_methods),
        "tests_with_more_than_two_to_five_focal_methods": len(
            tests_with_more_than_two_to_five_focal_methods
        ),
        "tests_with_more_than_five_to_ten_focal_methods": len(
            tests_with_more_than_five_to_ten_focal_methods
        ),
        "tests_with_more_than_ten_focal_methods": len(
            tests_with_more_than_ten_focal_methods
        ),
    }


def _create_summary(bucketed_dir: Path, results: List[Dict[str, Any] | None]) -> None:
    """Create a summary.json with test counts per project per bucket."""
    bucket_names = [
        "tests_with_one_focal_methods",
        "tests_with_two_focal_methods",
        "tests_with_more_than_two_to_five_focal_methods",
        "tests_with_more_than_five_to_ten_focal_methods",
        "tests_with_more_than_ten_focal_methods",
    ]
    totals = {bucket: 0 for bucket in bucket_names}
    totals["total"] = 0
    projects: Dict[str, Dict[str, int]] = {}

    for result in results:
        if result is None:
            continue

        dataset_name = result["dataset_name"]
        project_counts: Dict[str, int] = {}
        project_total = 0

        for bucket in bucket_names:
            count = result[bucket]
            project_counts[bucket] = count
            totals[bucket] += count
            project_total += count

        project_counts["total"] = project_total
        totals["total"] += project_total
        projects[dataset_name] = project_counts

    summary: Dict[str, Any] = {
        "projects": dict(sorted(projects.items())),
        "totals": totals,
    }

    with open(bucketed_dir / SUMMARY_FILE_NAME, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    """Process all projects and create bucketized datasets."""
    resources_path = ROOT_DIR / RESOURCES_DIR
    datasets_dir = resources_path / DATASETS_DIR
    hamster_dir = resources_path / HAMSTER_DIR
    bucketed_dir = resources_path / BUCKETED_DIR
    analysis_dir = resources_path / ANALYSIS_DIR

    hamster_dir.mkdir(parents=True, exist_ok=True)
    bucketed_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    projects = get_subfolders(hamster_dir)

    futures = []
    for project_folder in projects:
        project_name = Path(project_folder).name
        futures.append(
            _create_bucketized_dataset.remote(  # pyright: ignore[reportCallIssue]
                str(Path(project_folder) / INPUT_FILE_NAME),
                str(bucketed_dir / project_name),
                str(analysis_dir / project_name),
                str(datasets_dir / project_name),
            )
        )

    results: List[Dict[str, Any] | None] = []
    with tqdm(total=len(futures), desc="Processing projects...") as pbar:
        while futures:
            done, futures = ray.wait(futures, num_returns=1)
            res = ray.get(done)
            results.extend(res)
            pbar.update(len(done))

    _create_summary(bucketed_dir, results)


if __name__ == "__main__":
    main()
