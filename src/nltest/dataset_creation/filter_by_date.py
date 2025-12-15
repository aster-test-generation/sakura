import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ray
import tree_sitter_java as tsjava
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from cldk.analysis.java import JavaAnalysis
from tqdm import tqdm
from tree_sitter import Parser, Language

from nltest.utils.analysis.java_analyzer import CommonAnalysis

# Path constants
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_FILE_NAME = "nl2test.json"
SUMMARY_FILE_NAME = "summary.json"
DEFAULT_DATE_STR = "2025-01-31"

RESOURCES_DIR = "resources"
DATASETS_DIR = "datasets"
FILTERED_TESTS_DIR = "filtered_tests"
ANALYSIS_DIR = "analysis"


class FilterByDate:
    def __init__(self):
        self.java_lang: Language = Language(tsjava.language())
        self.parser: Parser = Parser(self.java_lang)


    def filter(self, repo_path: str, date_str: str):
        """Return all Java files added (first committed) after a given date."""
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
            "git", "-C", repo_path, "log",
            "--diff-filter=A", "--name-only", "--pretty=format:",
            f"--since={date_str}", "--", "*.java"
        ]

        result = subprocess.run(git_command, capture_output=True, text=True, check=True)
        files = list({line.strip() for line in result.stdout.split("\n") if line.strip()})
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

    def get_added_test_methods_treesitter(self, repo_path: str, date_str: str) -> Dict[str, List[str]]:
        """
        Detect new test methods added after a date using git diff + Tree-sitter.
        Returns { file_path -> [method_names] } where entire method is newly added.
        """
        git_diff_cmd = [
            "git", "-C", repo_path, "log", "--since", date_str,
            "-p", "--", "*.java"
        ]
        result = subprocess.run(git_diff_cmd, capture_output=True, text=True, check=True)

        new_tests: Dict[str, List[str]] = {}
        current_file = None
        added_lines: List[str] = []

        for line in result.stdout.splitlines():
            if line.startswith("+++ b/"):
                current_file = line[6:].strip()
                added_lines = []
                continue

            if not current_file:
                continue

            # Collect contiguous added lines only
            if line.startswith("+") and not line.startswith("+++"):
                added_lines.append(line[1:])

            # On hunk boundary or non-added lines: analyze block
            if (not line.startswith("+") and added_lines):
                self._analyze_added_block_with_treesitter(current_file, added_lines, new_tests)
                added_lines = []

        # Handle trailing block
        if added_lines:
            self._analyze_added_block_with_treesitter(current_file, added_lines, new_tests)

        for fpath, methods in new_tests.items():
            seen = set()
            new_tests[fpath] = [m for m in methods if not (m in seen or seen.add(m))]

        return new_tests

    def _analyze_added_block_with_treesitter(self, file_path: str, added_lines: List[str], new_tests: Dict[str, List[str]]):
        """
        Use Tree-sitter to extract @Test-annotated methods entirely within added block.
        """
        code_fragment = "\n".join(added_lines)
        if "@Test" not in code_fragment:
            return  # fast path — skip if no tests

        try:
            tree = self.parser.parse(bytes(code_fragment, "utf8"))
            root_node = tree.root_node

            def get_node_text(node):
                return code_fragment[node.start_byte:node.end_byte]

            for class_child in root_node.children:
                # descend to methods
                if class_child.type == "method_declaration":
                    method_code = get_node_text(class_child)
                    if "@Test" in method_code:
                        # full method body added
                        method_name = None
                        for n in class_child.children:
                            if n.type == "identifier":
                                method_name = get_node_text(n)
                                break
                        if method_name:
                            new_tests.setdefault(file_path, []).append(method_name)
        except Exception as e:
            print(f"[Tree-sitter] parse error in {file_path}: {e}")
            return

    def classify_files_using_cldk(
        self,
        file_list: List[str],
        repo_path: str,
        analysis: JavaAnalysis,
    ) -> Dict[str, List[str]]:
        """Classify using CLDK analysis for accurate test detection."""
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
                print("⚙️ Running Ray in local debug mode...")
                ray.init(local_mode=True, ignore_reinit_error=True)
            else:
                ray.init(ignore_reinit_error=True)
        if use_cldk and analysis_dir is None:
            raise ValueError("analysis_dir is required when use_cldk=True")

        futures = []
        for project_name in os.listdir(base_dir):
            repo_path = os.path.join(base_dir, project_name)
            if not os.path.isdir(repo_path) or not os.path.exists(os.path.join(repo_path, ".git")):
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
        total_new_test_methods = 0
        total_test_classes = 0
        total_test_methods = 0

        for item in collected_results:
            if item is None:
                continue
            project_name, project_result = item
            results[project_name] = project_result

            summary = project_result.get("summary", {})
            total_new_test_methods += summary.get("new_test_method_count", 0)
            total_test_classes += summary.get("test_class_count", 0)
            total_test_methods += summary.get("test_method_count", 0)

        results["summary"] = {
            "total_test_classes": total_test_classes,
            "total_test_methods": total_test_methods,
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
            json.dump({"summary": overall_summary, "projects": project_summaries}, f, indent=4)

        print(f"\n✅ Results saved to: {output_dir}")


RepoResult = Optional[Tuple[str, Dict]]


@ray.remote
def _process_single_repo(
    project_name: str,
    repo_path: str,
    date_str: str,
    use_cldk: bool,
    analysis_dir: Optional[str],
) -> RepoResult:
    """Process one repo."""
    filter_instance = FilterByDate()
    java_files = filter_instance.filter(repo_path, date_str)
    new_tests = filter_instance.get_added_test_methods_treesitter(repo_path, date_str)

    if use_cldk:
        try:
            repo_analysis_dir = Path(analysis_dir) / project_name
            analysis = CLDK(language="java").analysis(
                project_path=repo_path,
                analysis_backend_path=None,
                analysis_level=AnalysisLevel.symbol_table,
                analysis_json_path=repo_analysis_dir,
                eager=False,
            )
            test_classes_and_methods = filter_instance.classify_files_using_cldk(
                java_files, repo_path, analysis
            )
        except Exception as e:
            return project_name, {"error": str(e)}

        result = {
            "summary": {
                "test_class_count": len(test_classes_and_methods),
                "test_method_count": sum(len(m) for m in test_classes_and_methods.values()),
                "new_test_method_count": sum(len(m) for m in new_tests.values()),
            },
            "test_classes_and_methods": test_classes_and_methods,
            "new_test_methods_since_date": new_tests,
        }

    else:
        app_files, test_files = filter_instance.classify_files(java_files)
        result = {
            "summary": {
                "application_file_count": len(app_files),
                "test_file_count": len(test_files),
                "test_method_count": sum(filter_instance.count_tests_in_file(t, repo_path) for t in test_files),
                "new_test_method_count": sum(len(m) for m in new_tests.values()),
            },
            "application_files": app_files,
            "test_files": test_files,
            "new_test_methods_since_date": new_tests,
        }
    # --------------------------------------------------------------------------
    # Merge new test methods into test_classes_and_methods (avoid duplicates)
    # -------------------------------------------------------------------------

    merged = result["test_classes_and_methods"]

    for file_path, methods in result.get("new_test_methods_since_date", {}).items():
        normalized_methods = [m if m.endswith("()") else f"{m}()" for m in methods]

        # Default: derive simple class name
        class_name = os.path.splitext(os.path.basename(file_path))[0]
        fqcn_match = None

        # 1️⃣ Try to match existing CLDK FQCN by file path
        if analysis and merged:
            for fqcn in merged.keys():
                fqcn_path_fragment = fqcn.replace(".", os.sep)
                if fqcn_path_fragment in file_path:
                    fqcn_match = fqcn
                    break

        # 2️⃣ If no match, derive FQCN from path after src/test/java or src/main/java
        if not fqcn_match:
            lower_path = file_path.replace("\\", "/")
            if "src/test/java/" in lower_path:
                rel_path = lower_path.split("src/test/java/")[1]
                fqcn_match = rel_path.replace("/", ".").replace(".java", "")
            elif "src/main/java/" in lower_path:
                rel_path = lower_path.split("src/main/java/")[1]
                fqcn_match = rel_path.replace("/", ".").replace(".java", "")
            else:
                fqcn_match = None

        # 3️⃣ Fallback: simple class name
        key = fqcn_match if fqcn_match else class_name

        # Merge safely
        if key in merged:
            existing = set(merged[key])
            for m in normalized_methods:
                if m not in existing:
                    merged[key].append(m)
                    existing.add(m)
        else:
            merged[key] = list(dict.fromkeys(normalized_methods))

    # Deduplicate per class
    for cname, mlist in merged.items():
        seen = set()
        merged[cname] = [m for m in mlist if not (m in seen or seen.add(m))]

    # # Clean up and update summary
    # result["test_classes_and_methods"] = merged
    # result.pop("new_test_methods_since_date", None)
    #
    # result["summary"]["test_class_count"] = len(merged)
    # result["summary"]["test_method_count"] = sum(len(v) for v in merged.values())

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
        str(base_dir), date_str, use_cldk=use_cldk, analysis_dir=analysis_dir,
        debug=debug
    )
    filter_by_date.save_results(results, output_dir)


if __name__ == "__main__":
    main()
