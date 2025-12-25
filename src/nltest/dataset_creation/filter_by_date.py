import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ray
import tree_sitter_java as tsjava
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from cldk.analysis.java import JavaAnalysis
from tqdm import tqdm
from tree_sitter import Language, Node, Parser

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
    def __init__(self):
        self.java_lang: Language = Language(tsjava.language())
        self.parser: Parser = Parser(self.java_lang)

    def get_files_added_after_date(self, repo_path: str, date_str: str) -> List[str]:
        """Return Java files first committed after a given date."""
        try:
            subprocess.run(
                ["git", "-C", repo_path, "rev-parse", "--is-inside-work-tree"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            raise Exception(f"{repo_path} is not a valid Git repository.")

        git_command = [
            "git",
            "-C",
            repo_path,
            "log",
            "--diff-filter=A",
            "--name-only",
            "--pretty=format:",
            f"--since={date_str}",
            "--",
            "*.java",
        ]

        result = subprocess.run(git_command, capture_output=True, text=True, check=True)
        files = list(
            {line.strip() for line in result.stdout.split("\n") if line.strip()}
        )
        return files

    def classify_files(self, file_list: List[str]) -> Tuple[List[str], List[str]]:
        """Heuristically classify files as app or test."""
        app_files, test_files = [], []
        for file_path in file_list:
            is_test = (
                "/test/" in file_path.lower()
                or "\\test\\" in file_path.lower()
                or "Test" in os.path.basename(file_path)
            )
            (test_files if is_test else app_files).append(file_path)
        return app_files, test_files

    def count_tests_in_file(self, file_path, repo_path):
        """Count @Test annotations."""
        abs_path = os.path.join(repo_path, file_path)
        if not os.path.exists(abs_path):
            return 0
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                return len(re.findall(r"@Test\b", f.read()))
        except Exception:
            return 0

    def get_new_methods_via_blame(
        self, repo_path: str, date_str: str
    ) -> Dict[str, List[str]]:
        """Find methods added after a date by checking git blame timestamps."""
        new_methods: Dict[str, List[str]] = {}
        cutoff_timestamp = self._date_str_to_timestamp(date_str)

        test_files = self._find_candidate_test_files(repo_path)

        for file_path in test_files:
            abs_path = os.path.join(repo_path, file_path)
            if not os.path.exists(abs_path):
                continue

            try:
                with open(abs_path, "rb") as f:
                    content = f.read()
                tree = self.parser.parse(content)
            except Exception:
                continue

            line_timestamps = self._get_line_timestamps(repo_path, file_path)
            if not line_timestamps:
                continue

            methods = self._find_new_methods(
                tree.root_node,
                content.decode("utf-8", errors="ignore"),
                line_timestamps,
                cutoff_timestamp,
            )
            if methods:
                new_methods[file_path] = methods

        return new_methods

    def _date_str_to_timestamp(self, date_str: str) -> int:
        """Convert YYYY-MM-DD to Unix timestamp."""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp())

    def _find_candidate_test_files(self, repo_path: str) -> List[str]:
        """Find Java files that might contain tests."""
        result = subprocess.run(
            ["git", "-C", repo_path, "ls-files", "*.java"],
            capture_output=True,
            text=True,
        )
        files = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            if "/test/" in line.lower() or "Test" in os.path.basename(line):
                files.append(line)
        return files

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

    def _find_new_methods(
        self,
        root_node: Node,
        source_code: str,
        line_timestamps: Dict[int, int],
        cutoff_timestamp: int,
    ) -> List[str]:
        """Find methods where all lines were added after cutoff."""
        new_methods: List[str] = []

        def visit(node: Node) -> None:
            if node.type == "method_declaration":
                method_name = None
                for child in node.children:
                    if child.type == "identifier":
                        method_name = source_code[child.start_byte : child.end_byte]
                        break

                if not method_name:
                    return

                # Check if all lines were added after cutoff (1-indexed)
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                all_new = True
                for line_num in range(start_line, end_line + 1):
                    timestamp = line_timestamps.get(line_num, 0)
                    if timestamp < cutoff_timestamp:
                        all_new = False
                        break

                if all_new:
                    new_methods.append(method_name)
            else:
                for child in node.children:
                    visit(child)

        visit(root_node)
        return new_methods

    def extract_test_methods_from_files(
        self,
        file_list: List[str],
        repo_path: str,
        analysis: JavaAnalysis,
    ) -> Dict[str, List[str]]:
        """Extract test classes and methods from files using CLDK analysis."""
        common_analysis = CommonAnalysis(analysis)
        test_classes_and_methods: Dict[str, List[str]] = {}

        for rel_path in file_list:
            abs_path = os.path.join(repo_path, rel_path)
            compilation_unit = analysis.get_java_compilation_unit(abs_path)
            if not compilation_unit:
                continue

            for qualified_class_name in compilation_unit.type_declarations:
                frameworks = common_analysis.get_testing_frameworks_for_class(
                    qualified_class_name
                )
                if not frameworks:
                    continue

                test_methods = []
                for method_sig in analysis.get_methods_in_class(qualified_class_name):
                    if common_analysis.is_test_method(
                        method_sig, qualified_class_name, frameworks
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
        debug: bool = False,
    ) -> Dict:
        """Iterate through subdirectories and collect data."""
        if not ray.is_initialized():
            if debug:
                print("Running Ray in local debug mode...")
                ray.init(local_mode=True, ignore_reinit_error=True)
            else:
                ray.init(ignore_reinit_error=True)
        if use_cldk and analysis_dir is None:
            raise ValueError("analysis_dir is required when use_cldk=True")

        futures = []
        for project_name in os.listdir(base_dir):
            repo_path = os.path.join(base_dir, project_name)
            if not os.path.isdir(repo_path) or not os.path.exists(
                os.path.join(repo_path, ".git")
            ):
                continue
            analysis_dir_str = str(analysis_dir) if analysis_dir else None
            future = _process_single_repo.remote(
                project_name, repo_path, date_str, use_cldk, analysis_dir_str
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


def _normalize_method_sig(method_name: str, cldk_methods: List[str]) -> Optional[str]:
    """Find matching CLDK signature for a method name.

    Returns the full CLDK signature if found, None otherwise.
    """
    for sig in cldk_methods:
        if sig.split("(")[0] == method_name:
            return sig
    return None


def _find_fqcn_for_file(file_path: str, known_fqcns: List[str]) -> Optional[str]:
    """Find the FQCN that matches a file path."""
    for fqcn in known_fqcns:
        fqcn_path_fragment = fqcn.replace(".", os.sep)
        if fqcn_path_fragment in file_path:
            return fqcn

    # Derive FQCN from path after src/test/java or src/main/java
    normalized_path = file_path.replace("\\", "/")
    for marker in ("src/test/java/", "src/main/java/"):
        if marker in normalized_path:
            rel_path = normalized_path.split(marker)[1]
            return rel_path.replace("/", ".").replace(".java", "")

    return None


@ray.remote
def _process_single_repo(
    project_name: str,
    repo_path: str,
    date_str: str,
    use_cldk: bool,
    analysis_dir: Optional[str],
) -> RepoResult:
    """Process one repo to find tests added after a date."""
    filter_instance = FilterByDate()

    # Phase 1: Find completely new files added after the cutoff
    new_files = filter_instance.get_files_added_after_date(repo_path, date_str)

    # Phase 2: Find individual methods added after cutoff (in any test file)
    new_methods_via_blame = filter_instance.get_new_methods_via_blame(
        repo_path, date_str
    )

    if use_cldk:
        if analysis_dir is None:
            return project_name, {
                "error": "analysis_dir is required when use_cldk=True"
            }
        try:
            repo_analysis_dir = Path(analysis_dir) / project_name
            analysis = CLDK(language="java").analysis(
                project_path=repo_path,
                analysis_backend_path=None,
                analysis_level=AnalysisLevel.symbol_table,
                analysis_json_path=repo_analysis_dir,
                eager=False,
            )
            # Phase 3: Extract test methods from new files using CLDK
            test_classes_and_methods = filter_instance.extract_test_methods_from_files(
                new_files, repo_path, analysis
            )
        except Exception as e:
            return project_name, {"error": str(e)}

        # Phase 4: Process blame-detected methods from all test files (including existing)
        common_analysis = CommonAnalysis(analysis)
        verified_test_classes_and_methods: Dict[str, List[str]] = {}

        # First, add verified test methods from new files
        for class_name, method_sigs in test_classes_and_methods.items():
            testing_frameworks = common_analysis.get_testing_frameworks_for_class(
                class_name
            )
            if not testing_frameworks:
                continue

            verified_methods: List[str] = []
            for method_sig in method_sigs:
                if not analysis.get_method(class_name, method_sig):
                    continue
                if common_analysis.is_test_method(
                    method_sig, class_name, testing_frameworks
                ):
                    verified_methods.append(method_sig)

            if verified_methods:
                verified_test_classes_and_methods[class_name] = verified_methods

        # Then, process blame-detected methods (handles both new and existing files)
        for file_path, method_names in new_methods_via_blame.items():
            fqcn = _find_fqcn_for_file(file_path, [])
            if not fqcn:
                continue

            if not analysis.get_class(fqcn):
                continue

            testing_frameworks = common_analysis.get_testing_frameworks_for_class(fqcn)
            if not testing_frameworks:
                continue

            # Get all method signatures in this class for matching
            all_class_methods = list(analysis.get_methods_in_class(fqcn))

            existing_verified = set(verified_test_classes_and_methods.get(fqcn, []))
            new_verified: List[str] = []

            for method_name in method_names:
                method_sig = _normalize_method_sig(method_name, all_class_methods)
                if not method_sig:
                    continue
                if method_sig in existing_verified:
                    continue
                if not analysis.get_method(fqcn, method_sig):
                    continue
                if common_analysis.is_test_method(method_sig, fqcn, testing_frameworks):
                    new_verified.append(method_sig)
                    existing_verified.add(method_sig)

            if new_verified:
                if fqcn in verified_test_classes_and_methods:
                    verified_test_classes_and_methods[fqcn].extend(new_verified)
                else:
                    verified_test_classes_and_methods[fqcn] = new_verified

        result = {
            "summary": {
                "new_test_class_count": len(verified_test_classes_and_methods),
                "new_test_method_count": sum(
                    len(m) for m in verified_test_classes_and_methods.values()
                ),
            },
            "new_tests": verified_test_classes_and_methods,
        }

    else:
        # Non-CLDK path: use heuristics
        app_files, test_files = filter_instance.classify_files(new_files)
        result = {
            "summary": {
                "new_app_file_count": len(app_files),
                "new_test_file_count": len(test_files),
                "new_test_method_count": sum(
                    filter_instance.count_tests_in_file(t, repo_path)
                    for t in test_files
                ),
            },
            "new_app_files": app_files,
            "new_test_files": test_files,
        }

    return (project_name, result)


def main():
    resources_dir = ROOT_DIR / RESOURCES_DIR
    base_dir = resources_dir / DATASETS_DIR
    output_dir = resources_dir / FILTERED_TESTS_DIR
    date_str = DEFAULT_DATE_STR
    use_cldk = True
    debug = False
    analysis_dir = resources_dir / ANALYSIS_DIR

    filter_by_date = FilterByDate()
    results = filter_by_date.process_repos_in_dir(
        str(base_dir),
        date_str,
        use_cldk=use_cldk,
        analysis_dir=analysis_dir,
        debug=debug,
    )
    filter_by_date.save_results(results, output_dir)


if __name__ == "__main__":
    main()
