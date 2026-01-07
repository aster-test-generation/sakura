import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from cldk.analysis.java import JavaAnalysis
from tqdm import tqdm

from nltest.utils.analysis.java_analyzer import CommonAnalysis

# Path constants
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_FILE_NAME = "nl2test.json"
SUMMARY_FILE_NAME = "summary.json"
DEFAULT_DATE_STR = "2025-01-31"

RESOURCES_DIR = "resources"

# Relative to RESOURCES_DIR:
DATASETS_DIR = "datasets"
FILTERED_TESTS_DIR = "filtered_tests"
ANALYSIS_DIR = "analysis"


class FilterByDate:
    def _date_str_to_timestamp(self, date_str: str) -> int:
        """Convert YYYY-MM-DD to Unix timestamp."""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp())

    def _get_line_timestamps(self, repo_path: str, file_path: str) -> Dict[int, int]:
        """Get timestamp for each line using git blame."""
        cmd = ["git", "-C", repo_path, "blame", "--line-porcelain", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {}

        line_timestamps: Dict[int, int] = {}
        current_line = 0
        current_timestamp = 0

        for line in result.stdout.splitlines():
            if line.startswith("author-time "):
                current_timestamp = int(line.split()[1])
            elif line.startswith("\t"):
                current_line += 1
                line_timestamps[current_line] = current_timestamp

        return line_timestamps

    def get_new_methods_via_blame(
        self,
        repo_path: str,
        date_str: str,
        test_files: List[str],
        analysis: JavaAnalysis,
        test_class_methods: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """
        Find test methods added after a date by checking git blame timestamps.

        Args:
            repo_path: Path to the repository
            date_str: Cutoff date in YYYY-MM-DD format
            test_files: List of absolute paths to test files
            analysis: CLDK JavaAnalysis instance
            test_class_methods: Mapping of test class names to their test method signatures

        Returns:
            Mapping of qualified class names to lists of new test method signatures
        """
        cutoff_timestamp = self._date_str_to_timestamp(date_str)
        new_methods: Dict[str, List[str]] = {}

        for abs_file_path in test_files:
            rel_file_path = os.path.relpath(abs_file_path, repo_path)

            line_timestamps = self._get_line_timestamps(repo_path, rel_file_path)
            if not line_timestamps:
                continue

            file_new_methods = self._find_new_methods(
                abs_file_path,
                line_timestamps,
                cutoff_timestamp,
                analysis,
                test_class_methods,
            )
            for class_name, methods in file_new_methods.items():
                if class_name in new_methods:
                    new_methods[class_name].extend(methods)
                else:
                    new_methods[class_name] = methods

        return new_methods

    def _find_new_methods(
        self,
        file_path: str,
        line_timestamps: Dict[int, int],
        cutoff_timestamp: int,
        analysis: JavaAnalysis,
        test_class_methods: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """
        Find test methods where all lines were added after the cutoff date.

        Args:
            file_path: Absolute path to the Java file
            line_timestamps: Mapping of line numbers (1-indexed) to Unix timestamps
            cutoff_timestamp: Unix timestamp for the cutoff date
            analysis: CLDK JavaAnalysis instance
            test_class_methods: Mapping of test class names to their test method signatures

        Returns:
            Mapping of qualified class names to lists of new test method signatures
        """
        new_methods: Dict[str, List[str]] = {}

        compilation_unit = analysis.get_java_compilation_unit(file_path)
        if not compilation_unit:
            return new_methods

        for qualified_class_name in compilation_unit.type_declarations.keys():
            if qualified_class_name not in test_class_methods:
                continue

            valid_test_methods = set(test_class_methods[qualified_class_name])

            for method_signature in analysis.get_methods_in_class(qualified_class_name):
                if method_signature not in valid_test_methods:
                    continue

                method_details = analysis.get_method(
                    qualified_class_name, method_signature
                )
                if not method_details:
                    continue

                start_line = method_details.start_line
                end_line = method_details.end_line

                all_new = True
                for line_num in range(start_line, end_line + 1):
                    timestamp = line_timestamps.get(line_num, 0)
                    if timestamp < cutoff_timestamp:
                        all_new = False
                        break

                if all_new:
                    if qualified_class_name not in new_methods:
                        new_methods[qualified_class_name] = []
                    new_methods[qualified_class_name].append(method_signature)

        return new_methods

    def process_repos_in_dir(
        self,
        base_dir: str,
        date_str: str,
        analysis_dir: Path,
        debug: bool = False,
    ) -> Dict:
        """Iterate through subdirectories and collect data."""
        if not ray.is_initialized():
            if debug:
                print("Running Ray in local debug mode...")
                ray.init(local_mode=True, ignore_reinit_error=True)
            else:
                ray.init(ignore_reinit_error=True)

        futures = []
        for project_name in os.listdir(base_dir):
            repo_path = os.path.join(base_dir, project_name)
            if not os.path.isdir(repo_path) or not os.path.exists(
                os.path.join(repo_path, ".git")
            ):
                continue
            future = _process_single_repo.remote(
                project_name, repo_path, date_str, str(analysis_dir)
            )
            futures.append(future)

        collected_results = []
        with tqdm(total=len(futures), desc="Processing repositories") as pbar:
            while futures:
                done, futures = ray.wait(futures, num_returns=1)
                res = ray.get(done)
                collected_results.extend(res)
                pbar.update(len(done))

        results: Dict = {}
        total_new_test_classes = 0
        total_new_test_methods = 0

        for item in collected_results:
            if item is None:
                continue
            project_name, project_result = item
            results[project_name] = project_result

            summary = project_result.get("summary", {})
            total_new_test_classes += summary.get("new_test_class_count", 0)
            total_new_test_methods += summary.get("new_test_method_count", 0)

        results["summary"] = {
            "total_new_test_classes": total_new_test_classes,
            "total_new_test_methods": total_new_test_methods,
        }
        return results

    def save_results(self, results: Dict, output_dir: Path) -> None:
        """Save per-project and summary results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        overall_summary = results.pop("summary", {})
        project_summaries = {}

        for project_name, project_data in results.items():
            project_dir = output_dir / project_name
            project_dir.mkdir(parents=True, exist_ok=True)
            with open(project_dir / OUTPUT_FILE_NAME, "w", encoding="utf-8") as f:
                json.dump(project_data, f, indent=4)
            project_summaries[project_name] = project_data.get("summary", {})

        with open(output_dir / SUMMARY_FILE_NAME, "w", encoding="utf-8") as f:
            json.dump(
                {"summary": overall_summary, "projects": project_summaries}, f, indent=4
            )

        print(f"\nResults saved to: {output_dir}")


RepoResult = Optional[Tuple[str, Dict]]


@ray.remote
def _process_single_repo(
    project_name: str,
    repo_path: str,
    date_str: str,
    analysis_dir: str,
) -> RepoResult:
    """Process one repo to find tests added after a date."""
    filter_instance = FilterByDate()

    try:
        repo_analysis_dir = Path(analysis_dir) / project_name
        analysis = CLDK(language="java").analysis(
            project_path=repo_path,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=repo_analysis_dir,
            eager=False,
        )

        # Get all test classes and their test methods
        common_analysis = CommonAnalysis(analysis)
        test_class_methods, _, _ = common_analysis.categorize_classes()

        # Get test files from CLDK (absolute paths)
        test_files: Set[str] = set()
        for qualified_class_name in test_class_methods.keys():
            java_file = analysis.get_java_file(
                qualified_class_name=qualified_class_name
            )
            if java_file:
                test_files.add(java_file)

        # Find new methods via blame
        new_test_methods = filter_instance.get_new_methods_via_blame(
            repo_path, date_str, list(test_files), analysis, test_class_methods
        )

        result = {
            "summary": {
                "new_test_class_count": len(new_test_methods),
                "new_test_method_count": sum(len(m) for m in new_test_methods.values()),
            },
            "new_tests": new_test_methods,
        }

        return (project_name, result)

    except Exception as e:
        return (project_name, {"error": str(e)})


def main():
    resources_dir = ROOT_DIR / RESOURCES_DIR
    base_dir = resources_dir / DATASETS_DIR
    output_dir = resources_dir / FILTERED_TESTS_DIR
    date_str = DEFAULT_DATE_STR
    debug = False
    analysis_dir = resources_dir / ANALYSIS_DIR

    filter_by_date = FilterByDate()
    results = filter_by_date.process_repos_in_dir(
        str(base_dir),
        date_str,
        analysis_dir=analysis_dir,
        debug=debug,
    )
    filter_by_date.save_results(results, output_dir)


if __name__ == "__main__":
    main()
