import glob
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import ray
from bs4 import BeautifulSoup

from sakura.utils.analysis import CommonAnalysis
from sakura.javabuild.build_factory import BuildFactory
from sakura.utils.coverage.jacoco_test_watcher import TEST_CODE
from sakura.utils.pretty.color_logger import RichLog

JACOCO_CLI_JAR = (
    Path(__file__)
    .parent.parent.parent.parent.parent.joinpath("resources/lib/jacococli-0.8.13.jar")
    .absolute()
)
TEST_WATCHER_CLASS_NAME = "JaCoCoTestWatcher"
JACOCO_TEST_FOLDER = "target/jacoco-tests"


def _build_jacoco_report_command(
    project_root: Path, source_root: str, test: tuple[str, str]
) -> tuple[list[str], Path, Path]:
    project_root = project_root.absolute()
    class_name = test[0]
    method_name = test[1].split("(", maxsplit=1)[0].strip()
    exec_file = project_root.joinpath(
        JACOCO_TEST_FOLDER, f"{class_name}__{method_name}.exec"
    )
    report_dir = project_root.joinpath("target", f"{class_name}__{method_name}__report")
    source_path = Path(source_root)
    if not source_path.is_absolute():
        source_path = project_root.joinpath(source_path)

    command = [
        "java",
        "-jar",
        str(JACOCO_CLI_JAR),
        "report",
        str(exec_file),
        "--classfiles",
        str(project_root.joinpath("target", "classes")),
        "--sourcefiles",
        str(source_path),
        "--html",
        str(report_dir),
    ]
    return command, exec_file, report_dir


@ray.remote
def _collect_coverage_task(project_root: Path, source_root: str, test: Tuple[str, str]):
    jacococli_command, _, report_dir = _build_jacoco_report_command(
        project_root, source_root, test
    )
    try:
        RichLog.debug(f"Running command: {jacococli_command}")
        subprocess.run(
            jacococli_command,
            cwd=project_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        coverage_details = IndividualTestCoverage.extract_all_covered_lines(report_dir)
        return test, coverage_details, True
    except subprocess.CalledProcessError as e:
        RichLog.error(f'Error running command "{e.cmd}"')
        return test, e.stderr, False


class IndividualTestCoverage:
    def __init__(
        self,
        project_root: Path,
        target_module: str | None = None,
        build_type: str = "maven",
        source_root: str = "src/main/java",
        test_root: str = "src/test/java",
        junit_version: int = 5,
    ):
        self.project_root = project_root
        self.target_module = (
            target_module  # TODO: handle target module for multi-module projects
        )
        self.builder = BuildFactory.create(
            build_type, project_root, target_module=target_module
        )
        self.test_root = test_root
        self.app_source_path = (
            Path.cwd().joinpath(self.project_root, source_root).__str__()
        )
        self.junit_version = junit_version  # TODO: handle JUnit 4
        self.package_root = self.__get_root_package()
        self.source_root = source_root

    @classmethod
    def from_common_analysis(
        cls,
        common: CommonAnalysis,
        *,
        project_root: Path,
        qualified_class_names: List[str],
        module_root: Path | None = None,
        build_type: str = "maven",
        source_root: str = "src/main/java",
        junit_version: int = 5,
    ) -> "IndividualTestCoverage":
        resolved_module_root = Path(module_root) if module_root is not None else None
        if resolved_module_root is None:
            for class_name in qualified_class_names:
                candidate = common.resolve_module_root(class_name)
                if candidate is not None:
                    resolved_module_root = candidate
                    break

        if resolved_module_root is not None and not resolved_module_root.is_absolute():
            resolved_module_root = project_root / resolved_module_root

        resolved_project_root = (
            resolved_module_root if resolved_module_root is not None else project_root
        )
        test_base_dir = common.resolve_test_base_dir(
            resolved_module_root, project_root=project_root
        )
        return cls(
            project_root=resolved_project_root,
            build_type=build_type,
            source_root=source_root,
            test_root=str(test_base_dir),
            junit_version=junit_version,
        )

    def __get_root_package(self) -> str:
        """
        Computes and returns root package name for the application under test.

        Computes root package by finding the common path prefix of all directories under application
        source root that contain java source files.

        Returns:
            str: root package for app
        """
        app_dirs = []
        # ignored_java_files = ["package-info.java", "module-info.java"]
        for path, subdirs, files in os.walk(self.app_source_path):
            for d in subdirs:
                java_src_files = glob.glob(
                    os.path.join(path, d, "*.java"), recursive=False
                )
                if java_src_files:
                    app_dirs.append(os.path.join(path, d))
        common_path_prefix = os.path.commonprefix(app_dirs)
        root_package = (
            common_path_prefix.removeprefix(self.app_source_path)
            .strip(os.path.sep)
            .replace(os.path.sep, ".")
        )
        return root_package

    def generate(self, tests_to_run: list[tuple[str, str]] | None = None) -> dict:
        """
        Generate coverage for individual tests
        Args:
            tests_to_run: list of tests in (qualified_class_name, test_name) format

        Returns:
            dict[str, dict[str, dict]]
            key: Test class name
            value: key: test method name, value: coverage
        """
        tests_to_run = tests_to_run or []
        file_descriptor, build_file_name = tempfile.mkstemp(
            dir=self.project_root,
            prefix=".sakura-pom-cov-",
            suffix=".xml",
        )
        os.close(file_descriptor)
        modified_build_file = Path(build_file_name)
        original_class_content: dict[Path, bytes] = {}
        watcher_path = self.__get_watcher_path()
        watcher_snapshot: bytes | None = None
        watcher_was_snapshotted = False
        watcher_was_touched = False
        operation_error: BaseException | None = None
        try:
            all_test_classes = list(dict.fromkeys(test[0] for test in tests_to_run))
            test_class_name_path = IndividualTestCoverage.__get_test_class_info(
                test_root=str(self.project_root.joinpath(self.test_root)),
                qualified_names=all_test_classes,
            )
            original_class_content = self.__snapshot_test_classes(
                test_class_name_path, watcher_path
            )
            if watcher_path.exists():
                watcher_snapshot = watcher_path.read_bytes()
            watcher_was_snapshotted = True

            self.builder.add_code_coverage_dependencies(
                output_build_file=str(modified_build_file), add_java_agent=True
            )

            watcher_was_touched = True
            self.__add_helper_class_modify_tests(
                all_test_classes=test_class_name_path,
                original_class_content=original_class_content,
                watcher_path=watcher_path,
            )

            self.__run_tests(
                tests_to_run=tests_to_run,
                modified_build_file=modified_build_file,
                is_run_all_test=True if len(tests_to_run) == 0 else False,
            )

            coverage, exec_file_failures = self.__collect_coverage(
                executed_tests=(
                    tests_to_run
                    if len(tests_to_run)
                    else self.__get_executed_test_methods()
                )
            )
            RichLog.info(
                f"Coverage for each test method computed; jacoco report failures on files: {exec_file_failures}"
            )
            return coverage
        except BaseException as error:
            operation_error = error
            raise
        finally:
            cleanup_errors: list[Exception] = []
            try:
                cleanup_errors.extend(
                    self.__cleanup(
                        original_class_content=original_class_content,
                        watcher_path=watcher_path,
                        watcher_snapshot=(
                            watcher_snapshot if watcher_was_snapshotted else None
                        ),
                        restore_watcher=watcher_was_snapshotted and watcher_was_touched,
                    )
                )
            except Exception as error:
                cleanup_errors.append(error)

            try:
                modified_build_file.unlink(missing_ok=True)
            except Exception as error:
                cleanup_errors.append(error)

            if cleanup_errors:
                cleanup_error = ExceptionGroup(
                    "Individual test coverage cleanup failed", cleanup_errors
                )
                if operation_error is None:
                    raise cleanup_error
                operation_error.add_note(str(cleanup_error))

    def __collect_coverage(
        self, executed_tests: List[Tuple[str, str]]
    ) -> Tuple[Dict, List]:
        coverage = {}
        failing_exec_files = []
        # ray.init(num_cpus=100, ignore_reinit_error=True, include_dashboard=False)
        # ray_tasks = [
        #     _collect_coverage_task.remote(
        #         self.project_root, self.source_root, executed_test
        #     )
        #     for executed_test in executed_tests
        # ]
        # with tqdm(total=len(ray_tasks), desc="Collecting test coverage") as pbar:
        #     while ray_tasks:
        #         done, ray_tasks = ray.wait(ray_tasks, num_returns=1)
        #         test, test_coverage, success = ray.get(done[0])
        #         if success:
        #             if test[0] in coverage:
        #                 coverage[test[0]].append(
        #                     {
        #                         "test_class_name": test[0],
        #                         "test_name": test[1],
        #                         "coverage_details": test_coverage,
        #                     }
        #                 )
        #             else:
        #                 coverage[test[0]] = [
        #                     {
        #                         "test_class_name": test[0],
        #                         "test_name": test[1],
        #                         "coverage_details": test_coverage,
        #                     }
        #                 ]
        for test in executed_tests:
            method_name = test[1].split("(", maxsplit=1)[0].strip()
            jacococli_command, exec_file, report_dir = _build_jacoco_report_command(
                self.project_root, self.source_root, test
            )
            try:
                RichLog.info(f"Running command: {jacococli_command}")
                subprocess.run(
                    jacococli_command,
                    cwd=self.project_root,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                coverage_details = self.extract_all_covered_lines(report_dir)
                if test[0] in coverage:
                    coverage[test[0]].append(
                        {
                            "test_class_name": test[0],
                            "test_name": method_name,
                            "coverage_details": coverage_details,
                        }
                    )
                else:
                    coverage[test[0]] = [
                        {
                            "test_class_name": test[0],
                            "test_name": method_name,
                            "coverage_details": coverage_details,
                        }
                    ]
            except subprocess.CalledProcessError as e:
                RichLog.error(f'Error running command "{e.cmd}"')
                failing_exec_files.append(exec_file.name)
        RichLog.info(f"Coverage collected for {len(coverage.keys())} tests")
        return coverage, failing_exec_files

    @staticmethod
    def extract_all_covered_lines(jacoco_report_dir):
        result = {}
        pattern = os.path.join(jacoco_report_dir, "**", "*.html")
        html_files = glob.glob(pattern, recursive=True)
        for file in html_files:
            if "index.html" not in file and "jacoco-sessions" not in file:
                class_name, lines = (
                    IndividualTestCoverage.__extract_covered_lines_from_html(file)
                )
                if class_name and lines:
                    result[class_name] = lines
        return result

    @staticmethod
    def __extract_branch_lines_from_html(
        soup: BeautifulSoup,
    ) -> tuple[list[int], list[int], list[int]]:
        pre_block = soup.find("pre", class_="source")
        if not pre_block:
            return [], [], []

        lines = pre_block.find_all("span", id=re.compile(r"^L\d+"))

        covered = []
        missed = []
        partial = []

        for span in lines:
            line_num = int(span["id"][1:])
            classes = span.get("class", [])
            if "bfc" in classes:
                covered.append(line_num)
            elif "bnc" in classes:
                missed.append(line_num)
            elif "bpc" in classes:
                partial.append(line_num)

        return covered, missed, partial

    @staticmethod
    def __extract_covered_lines_from_html(html_path):
        if ".java.html" in html_path:
            with open(html_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")

            # Get fully qualified class name from breadcrumb
            breadcrumb = soup.select_one(".breadcrumb")
            if not breadcrumb:
                return None, None

            el_package = breadcrumb.select_one("a.el_package")
            if not el_package:
                return None, None
            el_class = breadcrumb.select_one("span.el_source")
            if not el_class:
                return None, None
            package_name = breadcrumb.select_one("a.el_package").text.strip()
            class_name = (
                breadcrumb.select_one("span.el_source")
                .text.strip()
                .replace(".java", "")
            )
            full_class_name = package_name + "." + class_name

            # Extract covered line numbers
            pre_block = soup.find("pre", class_="source")
            covered_lines = []
            if pre_block:
                for span in pre_block.find_all("span", id=re.compile(r"^L\d+")):
                    line_num = int(span["id"][1:])
                    cls = span.get("class", [])
                    if "fc" in cls or "pc" in cls:
                        covered_lines.append(line_num)
            covered_branches, uncovered_branches, partial_covered_branches = (
                IndividualTestCoverage.__extract_branch_lines_from_html(soup)
            )
            # Parse coverage summary table
            if Path(html_path.replace(".java.html", ".html")).exists():
                with open(
                    html_path.replace(".java.html", ".html"), "r", encoding="utf-8"
                ) as f:
                    soup = BeautifulSoup(f, "html.parser")
                covered_methods, uncovered_methods = (
                    IndividualTestCoverage.__extract_methods_with_coverage(soup)
                )
            else:
                covered_methods = uncovered_methods = 0
            if not covered_lines:
                return None, None

            return full_class_name, {
                "covered_lines": covered_lines,
                "method_covered": covered_methods,
                "method_missed": uncovered_methods,
                "branch_lines_covered": covered_branches,
                "branch_lines_partial": partial_covered_branches,
                "branch_lines_missed": uncovered_branches,
            }
        return None, None

    @staticmethod
    def __extract_methods_with_coverage(soup: BeautifulSoup) -> Tuple[list, list]:
        """
        Extract covered and uncovered methods
        Args:
            soup:

        Returns:

        """
        method_table = soup.find("table", class_="coverage")
        if not method_table:
            return [], []

        covered_methods = []
        missed_methods = []

        # Skip header row
        rows = method_table.find_all("tr")[1:]

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 3:
                continue

            method_cell = cols[0]
            coverage_cell = cols[2]  # This should be method coverage %

            method_link = method_cell.find("a", class_="el_method")
            if not method_link:
                continue

            method_name = method_link.get_text(strip=True)
            coverage_text = coverage_cell.get_text(strip=True).replace("%", "")

            try:
                coverage_percent = float(coverage_text)
                if coverage_percent > 0:
                    covered_methods.append(method_name)
                else:
                    missed_methods.append(method_name)
            except ValueError:
                missed_methods.append(method_name)

        return covered_methods, missed_methods

    def __run_tests(
        self,
        tests_to_run: list,
        modified_build_file: Path,
        is_run_all_test: bool = True,
        clean_jacoco_test_dir: bool = True,
    ):
        if clean_jacoco_test_dir:
            shutil.rmtree(
                self.project_root.joinpath(JACOCO_TEST_FOLDER), ignore_errors=True
            )

        target_tests = ""
        if not is_run_all_test:
            test_by_class = {}
            for test in tests_to_run:
                if test[0] not in test_by_class:
                    test_by_class[test[0]] = [test[1]]
                else:
                    test_by_class[test[0]].append(test[1])
            for test_class in test_by_class:
                if len(test_by_class[test_class]) == 1:
                    target_tests += (
                        test_class
                        + "#"
                        + test_by_class[test_class][0].split("(")[0]
                        + ","
                    )
                else:
                    target_tests += (
                        test_class + "#" + test_by_class[test_class][0].split("(")[0]
                    )
                    for i in range(1, len(test_by_class[test_class])):
                        target_tests += "+" + test_by_class[test_class][i].split("(")[0]
                    target_tests += ","
            if target_tests.endswith(","):
                target_tests = target_tests[:-1]

        self.builder.run_tests(
            target_tests=(target_tests if target_tests else None),
            build_file=str(modified_build_file),
        )

    @staticmethod
    def __cleanup(
        original_class_content: dict[Path, bytes],
        watcher_path: Path,
        watcher_snapshot: bytes | None,
        restore_watcher: bool,
    ) -> list[Exception]:
        cleanup_errors: list[Exception] = []
        for class_path, original_content in original_class_content.items():
            try:
                class_path.write_bytes(original_content)
            except Exception as error:
                cleanup_errors.append(error)

        if restore_watcher:
            try:
                if watcher_snapshot is None:
                    watcher_path.unlink(missing_ok=True)
                else:
                    watcher_path.write_bytes(watcher_snapshot)
            except Exception as error:
                cleanup_errors.append(error)

        return cleanup_errors

    def __get_watcher_path(self) -> Path:
        return self.project_root.joinpath(
            self.test_root,
            self.package_root.replace(".", os.sep),
            f"{TEST_WATCHER_CLASS_NAME}.java",
        )

    @staticmethod
    def __snapshot_test_classes(
        all_test_classes: list[tuple[str, str | None]], watcher_path: Path
    ) -> dict[Path, bytes]:
        original_class_content: dict[Path, bytes] = {}
        for _, test_class in all_test_classes:
            if test_class is None:
                continue
            test_path = Path(test_class)
            if test_path != watcher_path:
                original_class_content[test_path] = test_path.read_bytes()
        return original_class_content

    def __get_executed_test_methods(self) -> List[Tuple[str, str]]:
        executed_test_methods = []
        jacoco_test_dir = self.project_root.joinpath(JACOCO_TEST_FOLDER)
        if not jacoco_test_dir.exists():
            return executed_test_methods
        jacoco_exec_files: List[Path] = [
            f
            for f in jacoco_test_dir.iterdir()
            if f.is_file() and f.name.endswith(".exec")
        ]
        for exec_file in jacoco_exec_files:
            test_class_and_method = exec_file.name.removesuffix(".exec").split("__")
            executed_test_methods.append(
                (test_class_and_method[0], "__".join(test_class_and_method[1:]))
            )
        return executed_test_methods

    def __add_helper_class_modify_tests(
        self,
        all_test_classes: list[tuple[str, str | None]],
        original_class_content: dict[Path, bytes],
        watcher_path: Path,
    ) -> None:
        RichLog.info("Modifying helper code for individual test")
        watcher_path.parent.mkdir(parents=True, exist_ok=True)
        watcher_path.write_text(
            TEST_CODE.replace("<package_name>", self.package_root), encoding="utf-8"
        )

        for test_classes in all_test_classes:
            test_class = test_classes[1]
            if not test_class:
                RichLog.warn(f"Skipping missing test class path for {test_classes[0]}")
                continue
            test_path = Path(test_class)
            if test_path == watcher_path:
                continue
            test_class_content = original_class_content[test_path].decode("utf-8")
            if ".".join(test_classes[0].split(".")[:-1]) != self.package_root:
                test_class_content = self.__add_import(
                    test_class_content,
                    f"import {self.package_root}.{TEST_WATCHER_CLASS_NAME};",
                )
            if self.junit_version == 5:
                test_class_content = self.__add_import(
                    test_class_content,
                    "import org.junit.jupiter.api.extension.ExtendWith;",
                )
                test_class_content = self.__add_extends(
                    test_class_content, f"@ExtendWith({TEST_WATCHER_CLASS_NAME}.class)"
                )
            else:
                test_class_content = self.__add_import(
                    test_class_content, "import org.junit.runner.RunWith;"
                )
                test_class_content = self.__add_extends(
                    test_class_content, f"@RunWith({TEST_WATCHER_CLASS_NAME}.class)"
                )
            test_class_content = test_class_content.replace("@Disabled", "//@Disabled")
            test_path.write_text(test_class_content, encoding="utf-8")

    @staticmethod
    def __add_import(code: str, import_statement: str) -> str:
        """
        Adds import statements to the test file
        Args:
            code:
            import_statement:

        Returns:

        """
        if import_statement in code:
            return code  # Import already present

        # Find the last import statement
        matches = list(re.finditer(r"^import\s+.*?;", code, re.MULTILINE))
        if matches:
            last_import = matches[-1]
            insert_pos = last_import.end()
            return code[:insert_pos] + "\n" + import_statement + code[insert_pos:]
        else:
            # No import statements, insert after package if exists
            package_match = re.search(r"^package\s+.*?;", code, re.MULTILINE)
            if package_match:
                insert_pos = package_match.end()
                return code[:insert_pos] + "\n" + import_statement + code[insert_pos:]
            else:
                # No package or import, insert at top
                return import_statement + "\n" + code

    @staticmethod
    def __add_extends(code: str, annotation: str) -> str:
        """
        Adds an annotation like @ExtendWith(...) or @RunWith(...) before the class declaration.
        Special logic for @ExtendWith: it appends to existing list instead of duplicating.
        """
        is_extendwith = annotation.startswith("@ExtendWith(")
        class_match = re.search(
            r"^(\s*)(public\s+)?(class|interface|enum)\s+\w+", code, re.MULTILINE
        )

        if not class_match:
            return code  # No class found

        indent = class_match.group(1)
        new_class = IndividualTestCoverage.__extract_class_from_annotation(annotation)

        if is_extendwith:
            extend_match = re.search(r"@ExtendWith\s*\(\s*{?([^)}]*)}?[\s]*\)", code)
            if extend_match:
                existing_classes = [c.strip() for c in extend_match.group(1).split(",")]
                if new_class not in existing_classes:
                    existing_classes.append(new_class)
                    updated = (
                        f"@ExtendWith({{{', '.join(existing_classes)}}})"
                        if len(existing_classes) > 1
                        else f"@ExtendWith({existing_classes[0]})"
                    )
                    code = (
                        code[: extend_match.start()]
                        + updated
                        + code[extend_match.end() :]
                    )
                return code
            else:
                return IndividualTestCoverage.__insert_annotation(
                    code, annotation, indent, class_match.start()
                )
        else:
            if annotation in code:
                return code  # Annotation already exists
            return IndividualTestCoverage.__insert_annotation(
                code, annotation, indent, class_match.start()
            )

    @staticmethod
    def __insert_annotation(
        code: str, annotation: str, indent: str, position: int
    ) -> str:
        return code[:position] + f"{indent}{annotation}\n" + code[position:]

    @staticmethod
    def __extract_class_from_annotation(annotation: str) -> str:
        """
        From @ExtendWith(MyExtension.class), extract 'MyExtension.class'
        """
        match = re.match(r"@\w+\(\s*{?([^)}]*)}?[\s]*\)", annotation)
        if match:
            return match.group(1).split(",")[0].strip()
        return annotation  # fallback

    @staticmethod
    def __find_all_test_classes(test_root: str) -> list[tuple[str, str | None]]:
        """
        Walk through all Java files under test_root and return a list of:
        (fully_qualified_class_name, file_path)
        """
        results = []

        for root, _, files in os.walk(test_root):
            for file in files:
                if file.endswith(".java"):
                    full_path = os.path.join(root, file)

                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    # Extract package
                    package_match = re.search(
                        r"^\s*package\s+([\w.]+);", content, re.MULTILINE
                    )
                    if not package_match:
                        continue  # Skip if no package declaration

                    package = package_match.group(1)

                    # Extract public class name
                    class_match = re.search(
                        r"\b(public\s+)?(class|interface|enum)\s+(\w+)", content
                    )
                    if not class_match:
                        continue

                    class_name = class_match.group(3)
                    qualified_name = f"{package}.{class_name}"
                    results.append((qualified_name, full_path))

        return results

    @staticmethod
    def __resolve_class_paths(
        test_root: str, qualified_names: list[str]
    ) -> list[tuple[str, str | None]]:
        """
        Given a list of qualified names, return list of:
        (qualified_name, file_path)
        """
        results = []
        for qname in qualified_names:
            path = os.path.join(test_root, *qname.split(".")) + ".java"
            if os.path.isfile(path):
                results.append((qname, path))
            else:
                results.append((qname, None))  # File not found
        return results

    @staticmethod
    def __get_test_class_info(
        test_root: str, qualified_names: list[str] | None = None
    ) -> list[tuple[str, str | None]]:
        """
        Given a list of qualified test class name and their path
        Args:
            test_root:
            qualified_names:

        Returns:

        """
        if not qualified_names:
            return IndividualTestCoverage.__find_all_test_classes(test_root)
        else:
            return IndividualTestCoverage.__resolve_class_paths(
                test_root, qualified_names
            )
