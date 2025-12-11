import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cldk import CLDK
from cldk.analysis import AnalysisLevel
from cldk.analysis.java import JavaAnalysis

from nltest.utils.analysis.common_analysis import CommonAnalysis

# Path constants
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # Project root
OUTPUT_FILE_NAME = "nl2test.json"
SUMMARY_FILE_NAME = "summary.json"
RESOURCES_DIR = "resources"  # Relative to ROOT_DIR
DATASETS_DIR = "datasets"  # Relative to RESOURCES_DIR
FILTERED_TESTS_DIR = "filtered_tests"  # Relative to RESOURCES_DIR
ANALYSIS_DIR = "analysis"  # Relative to RESOURCES_DIR
DEFAULT_DATE_STR = "2025-01-31"


class FilterByDate:
    def __init__(self):
        pass

    def filter(self, repo_path: str, date_str: str):
        """
        Returns all Java files added (i.e., first committed) after a given date in a Git repo.

        Args:
            date_str (str): Date in 'YYYY-MM-DD' format.
            repo_path (str): Path to the Git repository (default current directory).

        Returns:
            list: List of Java file paths added after the given date.
        """
        try:
            # Verify it's a Git repo
            subprocess.run(
                ["git", "-C", repo_path, "rev-parse", "--is-inside-work-tree"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            raise Exception(f"{repo_path} is not a valid Git repository.")

            # Prepare the git log command
        git_command = [
            "git",
            "-C",
            repo_path,
            "log",
            "--diff-filter=A",  # Filter only added files
            "--name-only",  # Show only file names
            "--pretty=format:",  # Suppress commit headers
            f"--since={date_str}",  # Filter by date
            "--",
            "*.java",  # Only Java files
        ]

        # Execute the git command
        result = subprocess.run(git_command, capture_output=True, text=True, check=True)

        # Split output into individual files and remove duplicates
        files = list(
            {line.strip() for line in result.stdout.split("\n") if line.strip()}
        )

        return files

    def classify_files(self, file_list: List[str]) -> Tuple[List[str], List[str]]:
        """
        Categorize files into application and test files using path-based heuristics.

        Args:
            file_list: List of relative file paths.

        Returns:
            Tuple of (application_files, test_files).
        """
        application_files = []
        test_files = []

        for file_path in file_list:
            # Heuristic: check for /test/ directory or Test suffix in filename
            is_test = (
                "/test/" in file_path.lower()
                or "\\test\\" in file_path.lower()
                or "Test" in os.path.basename(file_path)
            )
            if is_test:
                test_files.append(file_path)
            else:
                application_files.append(file_path)

        return application_files, test_files

    def count_tests_in_file(self, file_path, repo_path):
        """
        Count number of test methods in a Java file (based on @Test annotations).
        """
        abs_path = os.path.join(repo_path, file_path)
        if not os.path.exists(abs_path):
            return 0
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                return len(re.findall(r"@Test\b", content))
        except Exception:
            return 0

    def classify_files_using_cldk(
        self,
        file_list: List[str],
        repo_path: str,
        analysis: JavaAnalysis,
    ) -> Dict[str, List[str]]:
        """
        Classify files using CLDK analysis to identify test classes and methods.

        Args:
            file_list: List of relative file paths from filter().
            repo_path: Path to the repository root.
            analysis: CLDK JavaAnalysis object for the repository.

        Returns:
            Dict mapping qualified test class names to list of test method signatures.
        """
        common_analysis = CommonAnalysis(analysis)
        test_classes_and_methods: Dict[str, List[str]] = {}

        # Process each file from the filter output
        for rel_path in file_list:
            abs_path = os.path.join(repo_path, rel_path)

            compilation_unit = analysis.get_java_compilation_unit(abs_path)

            if not compilation_unit:
                continue

            classes_in_file = compilation_unit.type_declarations

            for qualified_class_name in classes_in_file:
                testing_frameworks = common_analysis.get_testing_frameworks_for_class(
                    qualified_class_name
                )
                if not testing_frameworks:
                    continue

                # Check each method in the class
                test_methods = []
                for method_sig in analysis.get_methods_in_class(qualified_class_name):
                    if common_analysis.is_test_method(
                        method_sig, qualified_class_name, testing_frameworks
                    ):
                        test_methods.append(method_sig)

                if test_methods:
                    test_classes_and_methods[qualified_class_name] = test_methods

        return test_classes_and_methods

    def process_repos_in_dir(
        self,
        base_dir: str,
        date_str: str,
        use_cldk: bool = False,
        analysis_dir: Optional[Path] = None,
    ) -> Dict:
        """
        Iterate through subdirectories (each assumed to be a Git repo) and collect Java files added after date.

        Args:
            base_dir: Directory containing repository subdirectories.
            date_str: Date in 'YYYY-MM-DD' format to filter files added after.
            use_cldk: If True, use CLDK analysis for accurate test classification.
            analysis_dir: Directory for CLDK analysis JSON (required if use_cldk=True).

        Returns:
            Dict with per-repo results and overall summary.
            - CLDK mode: test_classes_and_methods mapping qualified names to method signatures.
            - Heuristic mode: application_files and test_files lists.
        """
        if use_cldk and analysis_dir is None:
            raise ValueError("analysis_dir is required when use_cldk=True")

        results: Dict = {}

        # Totals for CLDK mode
        total_test_classes = 0
        total_test_methods = 0

        # Totals for heuristic mode
        total_application_files = 0
        total_test_files = 0
        total_heuristic_test_methods = 0

        for project_name in os.listdir(base_dir):
            repo_path = os.path.join(base_dir, project_name)
            if not os.path.isdir(repo_path):
                continue

            if not os.path.exists(os.path.join(repo_path, ".git")):
                print(f"Skipping non-git directory: {repo_path}")
                continue

            print(f"Processing repository: {repo_path}")
            java_files = self.filter(repo_path=repo_path, date_str=date_str)

            if use_cldk:
                # Create CLDK analysis for this repo
                analysis_dir.mkdir(parents=True, exist_ok=True)
                repo_analysis_dir = analysis_dir / project_name
                analysis = CLDK(language="java").analysis(
                    project_path=repo_path,
                    analysis_backend_path=None,
                    analysis_level=AnalysisLevel.symbol_table,
                    analysis_json_path=repo_analysis_dir,
                    eager=False,
                )

                test_classes_and_methods = self.classify_files_using_cldk(
                    java_files, repo_path, analysis
                )

                repo_test_class_count = len(test_classes_and_methods)
                repo_test_method_count = sum(
                    len(methods) for methods in test_classes_and_methods.values()
                )

                total_test_classes += repo_test_class_count
                total_test_methods += repo_test_method_count

                results[project_name] = {
                    "summary": {
                        "test_class_count": repo_test_class_count,
                        "test_method_count": repo_test_method_count,
                    },
                    "test_classes_and_methods": test_classes_and_methods,
                }
            else:
                # Heuristic classification
                application_files, test_files = self.classify_files(java_files)

                app_file_count = len(application_files)
                test_file_count = len(test_files)
                repo_test_method_count = sum(
                    self.count_tests_in_file(t, repo_path) for t in test_files
                )

                total_application_files += app_file_count
                total_test_files += test_file_count
                total_heuristic_test_methods += repo_test_method_count

                results[project_name] = {
                    "summary": {
                        "application_file_count": app_file_count,
                        "test_file_count": test_file_count,
                        "test_method_count": repo_test_method_count,
                    },
                    "application_files": application_files,
                    "test_files": test_files,
                }

        # Add overall summary (different structure based on mode)
        if use_cldk:
            results["summary"] = {
                "total_test_classes": total_test_classes,
                "total_test_methods": total_test_methods,
            }
        else:
            results["summary"] = {
                "total_application_files": total_application_files,
                "total_test_files": total_test_files,
                "total_test_methods": total_heuristic_test_methods,
            }

        return results

    def save_results(self, results: Dict, output_dir: Path) -> None:
        """
        Save results to the output directory structure.

        Args:
            results: Results dict from process_repos_in_dir().
            output_dir: Directory to save output files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Separate overall summary from per-project results
        overall_summary = results.pop("summary", {})
        project_summaries: Dict[str, Dict] = {}

        # Write per-project files
        for project_name, project_data in results.items():
            project_dir = output_dir / project_name
            project_dir.mkdir(parents=True, exist_ok=True)

            project_file = project_dir / OUTPUT_FILE_NAME
            with open(project_file, "w", encoding="utf-8") as f:
                json.dump(project_data, f, indent=4)

            # Extract just the summary for the top-level file
            project_summaries[project_name] = project_data.get("summary", {})

        summary_file = output_dir / SUMMARY_FILE_NAME
        summary_data = {
            "summary": overall_summary,
            "projects": project_summaries,
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4)

        print(f"\nResults saved to: {output_dir}")
        print(f"  - Top-level summary: {summary_file}")
        print(f"  - Per-project data: {len(project_summaries)} projects")


def main():
    # Calculate paths relative to this file's location
    resources_dir = ROOT_DIR / RESOURCES_DIR
    base_dir = resources_dir / DATASETS_DIR
    output_dir = resources_dir / FILTERED_TESTS_DIR
    date_str = DEFAULT_DATE_STR
    use_cldk = True
    analysis_dir = resources_dir / ANALYSIS_DIR

    filter_by_date = FilterByDate()
    results = filter_by_date.process_repos_in_dir(
        str(base_dir), date_str, use_cldk=use_cldk, analysis_dir=analysis_dir
    )
    filter_by_date.save_results(results, output_dir)


if __name__ == "__main__":
    main()
