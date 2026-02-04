"""Compute per-project application code statistics for the dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple

import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from pydantic import ValidationError

from sakura.dataset_creation.model import NL2TestDataset
from sakura.utils.analysis.java_analyzer import CommonAnalysis
from sakura.utils.statistics import (
    DistributionSummary,
    distribution_to_dict,
    summarize_distribution,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = ROOT_DIR / "resources" / "analysis"
BUCKETED_DIR = ROOT_DIR / "resources" / "filtered_bucketed_tests"
DATASETS_DIR = ROOT_DIR / "resources" / "datasets"
OUTPUT_DIR = ROOT_DIR / "resources" / "filtered_bucketed_dataset_stats"
OUTPUT_FILE_NAME = "dataset_stats.json"
BUCKETED_FILE_NAME = "nl2test.json"


class ProjectStats(NamedTuple):
    project_name: str
    total_classes: int
    total_methods: int
    ncloc_distribution: DistributionSummary
    ncloc_values: List[float]
    focal_classes_distribution: DistributionSummary
    focal_classes_values: List[float]
    focal_methods_distribution: DistributionSummary
    focal_methods_values: List[float]
    total_tests: int
    error: str | None = None


def load_bucketed_dataset(project_name: str) -> NL2TestDataset | None:
    """Load the bucketed dataset for a project."""
    dataset_file = BUCKETED_DIR / project_name / BUCKETED_FILE_NAME
    if not dataset_file.exists():
        return None
    try:
        with dataset_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return NL2TestDataset.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        return None


def compute_focal_distributions(
    dataset: NL2TestDataset,
) -> tuple[List[float], List[float], int]:
    """Compute focal class and focal method counts per test.

    Returns:
        Tuple of (focal_classes_per_test, focal_methods_per_test, total_tests)
    """
    focal_classes_counts: List[float] = []
    focal_methods_counts: List[float] = []

    buckets = [
        dataset.tests_with_one_focal_methods,
        dataset.tests_with_two_focal_methods,
        dataset.tests_with_more_than_two_to_five_focal_methods,
        dataset.tests_with_more_than_five_to_ten_focal_methods,
        dataset.tests_with_more_than_ten_focal_methods,
    ]

    for tests in buckets:
        for test in tests:
            if test.focal_details is None:
                focal_classes_counts.append(0)
                focal_methods_counts.append(0)
                continue

            num_focal_classes = len(test.focal_details)
            num_focal_methods = sum(
                len(fc.focal_method_names or []) for fc in test.focal_details
            )
            focal_classes_counts.append(float(num_focal_classes))
            focal_methods_counts.append(float(num_focal_methods))

    total_tests = len(focal_classes_counts)
    return focal_classes_counts, focal_methods_counts, total_tests


@ray.remote
def process_project(project_name: str) -> ProjectStats:
    """Process a single project and compute its statistics."""
    analysis_path = ANALYSIS_DIR / project_name
    analysis_json = analysis_path / "analysis.json"
    project_path = DATASETS_DIR / project_name

    if not analysis_json.exists():
        return ProjectStats(
            project_name=project_name,
            total_classes=0,
            total_methods=0,
            ncloc_distribution=summarize_distribution([]),
            ncloc_values=[],
            focal_classes_distribution=summarize_distribution([]),
            focal_classes_values=[],
            focal_methods_distribution=summarize_distribution([]),
            focal_methods_values=[],
            total_tests=0,
            error=f"Missing analysis.json at {analysis_json}",
        )

    if not project_path.exists():
        return ProjectStats(
            project_name=project_name,
            total_classes=0,
            total_methods=0,
            ncloc_distribution=summarize_distribution([]),
            ncloc_values=[],
            focal_classes_distribution=summarize_distribution([]),
            focal_classes_values=[],
            focal_methods_distribution=summarize_distribution([]),
            focal_methods_values=[],
            total_tests=0,
            error=f"Missing project directory at {project_path}",
        )

    try:
        analysis = CLDK(language="java").analysis(
            project_path=str(project_path),
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=str(analysis_path),
            eager=False,
        )
    except Exception as e:
        return ProjectStats(
            project_name=project_name,
            total_classes=0,
            total_methods=0,
            ncloc_distribution=summarize_distribution([]),
            ncloc_values=[],
            focal_classes_distribution=summarize_distribution([]),
            focal_classes_values=[],
            focal_methods_distribution=summarize_distribution([]),
            focal_methods_values=[],
            total_tests=0,
            error=f"Failed to load analysis: {e}",
        )

    common_analysis = CommonAnalysis(analysis)

    # Categorize classes to get application classes
    _, application_classes, _ = common_analysis.categorize_classes()

    total_classes = len(application_classes)
    total_methods = 0
    ncloc_values: List[float] = []

    for qualified_class_name in application_classes:
        methods = analysis.get_methods_in_class(qualified_class_name)
        total_methods += len(methods)

        for method_sig in methods:
            method_details = analysis.get_method(qualified_class_name, method_sig)
            if method_details:
                ncloc = common_analysis.get_ncloc(
                    method_details.declaration, method_details.code
                )
                ncloc_values.append(float(ncloc))

    ncloc_distribution = summarize_distribution(ncloc_values)

    # Load bucketed dataset for focal distributions
    dataset = load_bucketed_dataset(project_name)
    if dataset:
        focal_classes_counts, focal_methods_counts, total_tests = (
            compute_focal_distributions(dataset)
        )
        focal_classes_distribution = summarize_distribution(focal_classes_counts)
        focal_methods_distribution = summarize_distribution(focal_methods_counts)
    else:
        focal_classes_counts = []
        focal_methods_counts = []
        focal_classes_distribution = summarize_distribution([])
        focal_methods_distribution = summarize_distribution([])
        total_tests = 0

    return ProjectStats(
        project_name=project_name,
        total_classes=total_classes,
        total_methods=total_methods,
        ncloc_distribution=ncloc_distribution,
        ncloc_values=ncloc_values,
        focal_classes_distribution=focal_classes_distribution,
        focal_classes_values=focal_classes_counts,
        focal_methods_distribution=focal_methods_distribution,
        focal_methods_values=focal_methods_counts,
        total_tests=total_tests,
    )


def project_stats_to_dict(stats: ProjectStats) -> Dict[str, Any]:
    """Convert ProjectStats to a dictionary for JSON serialization."""
    result: Dict[str, Any] = {
        "project_name": stats.project_name,
        "total_application_classes": stats.total_classes,
        "total_application_methods": stats.total_methods,
        "total_application_ncloc": stats.ncloc_distribution.total,
        "ncloc_distribution": distribution_to_dict(stats.ncloc_distribution),
        "focal_classes_distribution": distribution_to_dict(
            stats.focal_classes_distribution
        ),
        "focal_methods_distribution": distribution_to_dict(
            stats.focal_methods_distribution
        ),
        "total_tests": stats.total_tests,
    }
    if stats.error:
        result["error"] = stats.error
    return result


def main() -> None:
    if not ANALYSIS_DIR.exists():
        print(f"Analysis directory not found: {ANALYSIS_DIR}")
        return

    # Get list of projects from analysis directory
    project_dirs = [
        path.name for path in sorted(ANALYSIS_DIR.iterdir()) if path.is_dir()
    ]

    if not project_dirs:
        print(f"No projects found in {ANALYSIS_DIR}")
        return

    print(f"Processing {len(project_dirs)} projects...")

    ray.init(ignore_reinit_error=True)

    try:
        tasks = [process_project.remote(project_name) for project_name in project_dirs]
        results: List[ProjectStats] = ray.get(tasks)
    finally:
        ray.shutdown()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Aggregate statistics for summary
    total_app_classes = 0
    total_app_methods = 0
    total_tests_all = 0
    all_ncloc_values: List[float] = []
    all_focal_classes: List[float] = []
    all_focal_methods: List[float] = []
    successful_projects = 0
    failed_projects = 0
    no_dataset_projects = 0
    included_results: List[ProjectStats] = []

    for stats in results:
        if stats.error:
            print(f"[FAILED] {stats.project_name}: {stats.error}")
            failed_projects += 1
            continue

        if stats.total_tests == 0:
            print(f"[NO DATASET] {stats.project_name}: no bucketed tests found")
            no_dataset_projects += 1
            continue

        # Write per-project stats only for projects with tests
        project_output_dir = OUTPUT_DIR / stats.project_name
        project_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = project_output_dir / OUTPUT_FILE_NAME

        stats_dict = project_stats_to_dict(stats)
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(stats_dict, handle, indent=2)

        print(
            f"[OK] {stats.project_name}: {stats.total_classes} classes, "
            f"{stats.total_methods} methods, {stats.total_tests} tests"
        )
        successful_projects += 1
        total_app_classes += stats.total_classes
        total_app_methods += stats.total_methods
        total_tests_all += stats.total_tests

        all_ncloc_values.extend(stats.ncloc_values)
        all_focal_classes.extend(stats.focal_classes_values)
        all_focal_methods.extend(stats.focal_methods_values)
        included_results.append(stats)

    # Build summary
    ncloc_distribution = summarize_distribution(all_ncloc_values)
    summary: Dict[str, Any] = {
        "total_projects": len(project_dirs),
        "successful_projects": successful_projects,
        "failed_projects": failed_projects,
        "no_dataset_projects": no_dataset_projects,
        "total_application_classes": total_app_classes,
        "total_application_methods": total_app_methods,
        "total_application_ncloc": ncloc_distribution.total,
        "total_tests": total_tests_all,
        "aggregate_ncloc_distribution": distribution_to_dict(ncloc_distribution),
        "aggregate_focal_classes_distribution": distribution_to_dict(
            summarize_distribution(all_focal_classes)
        ),
        "aggregate_focal_methods_distribution": distribution_to_dict(
            summarize_distribution(all_focal_methods)
        ),
        "projects": [project_stats_to_dict(stats) for stats in included_results],
    }

    summary_file = OUTPUT_DIR / "summary.json"
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nSummary:")
    print(f"  Total projects: {len(project_dirs)}")
    print(f"  Successful: {successful_projects}")
    print(f"  Failed: {failed_projects}")
    print(f"  No dataset: {no_dataset_projects}")
    print(f"  Total application classes: {total_app_classes}")
    print(f"  Total application methods: {total_app_methods}")
    print(f"  Total application NCLOC: {int(ncloc_distribution.total)}")
    print(f"  Total tests: {total_tests_all}")
    print(f"\nResults written to: {OUTPUT_DIR}")
    print(f"Summary written to: {summary_file}")


if __name__ == "__main__":
    main()
