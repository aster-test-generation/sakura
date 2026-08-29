import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

from sakura.javabuild.maven_build import MavenBuild
from pydantic import BaseModel

from sakura.utils.analysis import CommonAnalysis


class ExecutionIssue(BaseModel):
    """Single failure or error captured from a Maven test run."""

    class_name: str
    test_name: str
    kind: str  # "failure" or "error"
    message: Optional[str] = None
    error_type: Optional[str] = None
    file: Optional[str] = None
    line: Optional[int] = None
    stack_trace: Optional[str] = None


class JavaMavenExecution(MavenBuild):
    """Run Maven tests and parse failures/errors from Surefire or console logs."""

    HEAD_STACKTRACE_LINES = 8
    TAIL_STACKTRACE_LINES = 4
    FAILURE_HEADER_RE = re.compile(
        r"^\[ERROR\]\s+(?:"
        r"(?P<class>[A-Za-z0-9_.$]+)\.(?P<test>[^\s]+)"  # com.example.MyTest.testMethod
        r"|"
        r"(?P<test2>[^\s(]+)\((?P<class2>[A-Za-z0-9_.$]+)\)"  # testMethod(com.example.MyTest)
        r")\s+.*<<<\s+(?P<kind>FAILURE|ERROR)!"
    )
    FAILURE_SUMMARY_RE = re.compile(
        r"^\[ERROR\]\s+(?P<class>[A-Za-z0-9_.$]+)\.(?P<test>[^:]+?)(?::(?P<line>\d+))?\s+(?P<message>.+)$"
    )
    STACK_FILE_RE = re.compile(r"\(([^()/:\\\s]+\.java):(\d+)\)")
    STACK_FALLBACK_FILE_RE = re.compile(r"([A-Za-z0-9_\-./\\]+\.java):(\d+)")

    def __init__(self, project_root: Path, module_root: Path | None = None):
        # Expects absolute project_root and module_root
        # MavenBuild expects target_module to be relative to project_root
        project_abs = project_root.resolve()
        module_abs = module_root.resolve() if module_root is not None else None

        target_module = None
        if module_abs and project_abs != module_abs:
            try:
                rel_str = module_abs.relative_to(project_abs).as_posix()
                if rel_str not in ("", "."):
                    target_module = rel_str
            except ValueError:
                target_module = None

        if target_module:
            super().__init__(str(project_abs), target_module=target_module)
        else:
            super().__init__(str(project_abs))

        self.module_root = module_abs or project_abs

    def get_execution_errors(
        self,
        qualified_class_name: Optional[str] = None,
        method_signature: Optional[str] = None,
        prepare_dependencies: bool = False,
        timeout: int | None = 600,
    ) -> List[ExecutionIssue]:
        if prepare_dependencies and self.target_module:
            self.install_selected_projects_skip_tests_proc(timeout=timeout)

        target_tests: Optional[str] = None
        expected_class: Optional[str] = qualified_class_name
        expected_method: Optional[str] = None

        if qualified_class_name:
            simple_class = CommonAnalysis.get_simple_class_name(qualified_class_name)
            if method_signature:
                expected_method = CommonAnalysis.get_simple_method_name(
                    method_signature
                )
                target_tests = f"{simple_class}#{expected_method}"
            else:
                target_tests = simple_class

        run_started_at = time.time() - 2.0
        proc = self.run_tests_proc(
            target_tests=target_tests, also_make=False, timeout=timeout
        )
        runtime_output = proc.stdout

        report_dirs = self.find_surefire_reports()
        matched_any = False
        if report_dirs and expected_class:
            issues, matched_any = self.parse_surefire_reports_for_target(
                report_dirs,
                min_mtime=run_started_at,
                expected_class=expected_class,
                expected_method=expected_method,
            )
            if issues:
                return issues

        console_issues = self.parse_console_output(runtime_output)
        if console_issues:
            return console_issues

        if proc.returncode != 0:
            return [
                self._maven_process_issue(
                    returncode=proc.returncode,
                    stdout=runtime_output,
                    expected_class=expected_class,
                    expected_method=expected_method,
                )
            ]

        if matched_any:
            return []

        # If we targeted a test but have no evidence it ran, return an explicit error
        if expected_class and target_tests:
            return [
                ExecutionIssue(
                    class_name=expected_class,
                    test_name=expected_method or "",
                    kind="error",
                    error_type="NoSpecifiedTests",
                    message="Specified test was not executed (no matching surefire testcase found).",
                )
            ]

        return []

    def find_surefire_reports(self) -> List[str]:
        """Locate every target/surefire-reports directory in the project tree."""

        found: List[str] = []

        search_root = self.module_root

        # Fast path: the canonical surefire location for the module you just tested
        direct = search_root / "target" / "surefire-reports"
        if direct.is_dir():
            return [str(direct)]

        # Fallback: scan subtree (covers unusual layouts)
        for dirpath, dirnames, _ in os.walk(str(search_root)):
            if "surefire-reports" in dirnames and os.path.basename(dirpath) == "target":
                found.append(os.path.join(dirpath, "surefire-reports"))

        return found

    def parse_surefire_reports_for_target(
        self,
        reports_dirs: List[str],
        min_mtime: Optional[float] = None,
        expected_class: Optional[str] = None,
        expected_method: Optional[str] = None,
    ) -> Tuple[List[ExecutionIssue], bool]:
        issues: List[ExecutionIssue] = []
        matched_any = False

        for rdir in reports_dirs:
            try:
                files = [
                    f
                    for f in os.listdir(rdir)
                    if f.startswith("TEST-") and f.endswith(".xml")
                ]
            except FileNotFoundError:
                continue

            for file_name in files:
                file_path = os.path.join(rdir, file_name)

                if min_mtime is not None:
                    try:
                        if os.path.getmtime(file_path) < min_mtime:
                            continue
                    except OSError:
                        pass

                try:
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                except Exception:
                    continue

                # Normalize to a list of <testsuite>
                suites: List[ET.Element] = []
                root_name = self._local_name(root.tag)
                if root_name == "testsuite":
                    suites = [root]
                elif root_name == "testsuites":
                    suites = [
                        child
                        for child in root
                        if self._local_name(child.tag) == "testsuite"
                    ]
                else:
                    suites = [
                        child
                        for child in root.iter()
                        if self._local_name(child.tag) == "testsuite"
                    ]

                for suite in suites:
                    for tc in (
                        child
                        for child in suite
                        if self._local_name(child.tag) == "testcase"
                    ):
                        class_name = tc.get("classname") or ""
                        test_name = tc.get("name") or ""

                        if expected_class and class_name != expected_class:
                            continue
                        if expected_method:
                            if not (
                                test_name == expected_method
                                or test_name.startswith(f"{expected_method}(")
                                or test_name.startswith(f"{expected_method}[")
                            ):
                                continue

                        matched_any = True

                        for kind_tag in ("failure", "error"):
                            for element in (
                                child
                                for child in tc
                                if self._local_name(child.tag) == kind_tag
                            ):
                                issue = ExecutionIssue(
                                    class_name=class_name,
                                    test_name=test_name,
                                    kind="failure"
                                    if kind_tag == "failure"
                                    else "error",
                                    message=element.get("message"),
                                    error_type=element.get("type"),
                                    stack_trace=self._truncate_stack_trace(
                                        (element.text or "").strip() or None
                                    ),
                                )
                                self._maybe_fill_file_line(issue)
                                issues.append(issue)

        return issues, matched_any

    @classmethod
    def _maven_process_issue(
        cls,
        returncode: int,
        stdout: str | None,
        expected_class: str | None,
        expected_method: str | None,
    ) -> ExecutionIssue:
        if returncode == -1:
            error_type = "MavenTimeout"
            message = "Maven test command timed out."
        else:
            error_type = "MavenBuildFailure"
            message = f"Maven test command exited with status {returncode}."

        output = cls._concise_output(stdout)
        if output:
            message = f"{message} Output: {output}"

        return ExecutionIssue(
            class_name=expected_class or "",
            test_name=expected_method or "",
            kind="error",
            error_type=error_type,
            message=message,
        )

    @staticmethod
    def _concise_output(stdout: str | None, max_chars: int = 500) -> str | None:
        if not stdout:
            return None

        output = " ".join(stdout.split())
        if len(output) > max_chars:
            return f"...{output[-(max_chars - 3) :]}"
        return output or None

    def parse_surefire_reports(
        self,
        reports_dirs: List[str],
        min_mtime: Optional[float] = None,
    ) -> List[ExecutionIssue]:
        """Read Surefire XML reports and convert every failure/error into ExecutionIssue objects."""

        issues: List[ExecutionIssue] = []
        for rdir in reports_dirs:
            try:
                files = [
                    f
                    for f in os.listdir(rdir)
                    if f.startswith("TEST-") and f.endswith(".xml")
                ]
            except FileNotFoundError:
                continue

            for file_name in files:
                file_path = os.path.join(rdir, file_name)

                if min_mtime is not None:
                    try:
                        if os.path.getmtime(file_path) < min_mtime:
                            continue
                    except OSError:
                        pass

                try:
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                except Exception:
                    continue

                suites: List[ET.Element] = []
                root_name = self._local_name(root.tag)
                if root_name == "testsuite":
                    suites = [root]
                elif root_name == "testsuites":
                    suites = [
                        child
                        for child in root
                        if self._local_name(child.tag) == "testsuite"
                    ]
                else:
                    # Fallback: treat any <testsuite> in the document as a suite
                    suites = [
                        child
                        for child in root.iter()
                        if self._local_name(child.tag) == "testsuite"
                    ]

                for suite in suites:
                    for tc in (
                        child
                        for child in suite
                        if self._local_name(child.tag) == "testcase"
                    ):
                        class_name = tc.get("classname") or ""
                        test_name = tc.get("name") or ""

                        for kind_tag in ("failure", "error"):
                            for element in (
                                child
                                for child in tc
                                if self._local_name(child.tag) == kind_tag
                            ):
                                issue = ExecutionIssue(
                                    class_name=class_name,
                                    test_name=test_name,
                                    kind="failure"
                                    if kind_tag == "failure"
                                    else "error",
                                    message=element.get("message"),
                                    error_type=element.get("type"),
                                    stack_trace=self._truncate_stack_trace(
                                        (element.text or "").strip() or None
                                    ),
                                )
                                self._maybe_fill_file_line(issue)
                                issues.append(issue)

        return issues

    def parse_console_output(self, stdout: Optional[str]) -> List[ExecutionIssue]:
        """Best-effort parsing of Maven console logs when Surefire XML is unavailable."""

        if not stdout:
            return []

        issues: List[ExecutionIssue] = []
        lines = stdout.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            header = self.FAILURE_HEADER_RE.match(line)
            summary = None if header else self.FAILURE_SUMMARY_RE.match(line)

            if header or summary:
                if header:
                    # Support both com.example.MyTest.testMethod and testMethod(com.example.MyTest)
                    class_name = (
                        header.group("class") or header.group("class2")
                    ).strip()
                    test_name = (header.group("test") or header.group("test2")).strip()
                    kind = (
                        "failure"
                        if header.group("kind").upper() == "FAILURE"
                        else "error"
                    )
                    message = None
                    line_no = None
                else:
                    class_name = summary.group("class")
                    test_name = summary.group("test").strip()
                    line_str = summary.group("line")
                    line_no = int(line_str) if line_str else None
                    message = summary.group("message").strip()
                    # Heuristic classification for summary-only lines
                    lowered = message.lower()
                    if "error" in lowered or "exception" in lowered:
                        kind = "error"
                    else:
                        kind = "failure"

                stack_trace, next_index = self._collect_stack_trace(lines, i + 1)
                error_type, derived_message = self._extract_error_type_and_message(
                    stack_trace
                )
                if not message and derived_message:
                    message = derived_message

                issue = ExecutionIssue(
                    class_name=class_name,
                    test_name=test_name,
                    kind=kind,
                    message=message,
                    error_type=error_type,
                    line=line_no,
                    stack_trace=stack_trace,
                )
                self._maybe_fill_file_line(issue)
                issues.append(issue)
                i = next_index
                continue

            i += 1

        return issues

    @classmethod
    def _collect_stack_trace(
        cls, lines: List[str], start_index: int
    ) -> Tuple[Optional[str], int]:
        collected: List[str] = []
        idx: int = start_index
        blank_streak = 0

        while idx < len(lines):
            current: str = lines[idx]
            if cls.FAILURE_HEADER_RE.match(current) or cls.FAILURE_SUMMARY_RE.match(
                current
            ):
                break

            stripped: str = current.strip()
            if (
                stripped.startswith("[INFO] Results")
                or stripped.startswith("[INFO] Tests run")
                or stripped.startswith("[ERROR] Results")
                or stripped.startswith("[ERROR] Tests run")
                or "There are test failures" in stripped
            ):
                break
            if (
                stripped.startswith("[INFO] BUILD")
                or stripped.startswith("[ERROR] BUILD")
                or stripped.startswith("[INFO] ---")
                or stripped.startswith("[ERROR] ---")
            ):
                break

            entry = current
            if current.startswith("[ERROR]"):
                entry = current[len("[ERROR]") :].lstrip()
            elif current.startswith("[INFO]"):
                entry = current[len("[INFO]") :].lstrip()

            if not entry:
                blank_streak += 1
                if blank_streak > 1 and collected:
                    idx += 1
                    break
            else:
                blank_streak = 0

            if entry or collected:
                collected.append(entry)

            idx += 1

        stack = "\n".join(collected).strip() if collected else None
        stack = cls._truncate_stack_trace(stack)
        return stack, idx

    @classmethod
    def _maybe_fill_file_line(cls, issue: ExecutionIssue) -> None:
        if issue.line is not None and issue.file:
            return
        if not issue.stack_trace:
            return

        match = cls.STACK_FILE_RE.search(issue.stack_trace)
        if not match:
            match = cls.STACK_FALLBACK_FILE_RE.search(issue.stack_trace)
        if match:
            issue.file = match.group(1)
            try:
                issue.line = int(match.group(2))
            except ValueError:
                issue.line = None

    @classmethod
    def _truncate_stack_trace(cls, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        lines = text.splitlines()
        total = len(lines)
        max_lines = cls.HEAD_STACKTRACE_LINES + cls.TAIL_STACKTRACE_LINES
        if total <= max_lines:
            return text

        skipped = total - max_lines
        head = lines[: cls.HEAD_STACKTRACE_LINES]
        tail = lines[-cls.TAIL_STACKTRACE_LINES :]

        return (
            "\n".join(head)
            + f"\n    ... (skipped {skipped} lines) ...\n"
            + "\n".join(tail)
        )

    @staticmethod
    def _extract_error_type_and_message(
        stack_trace: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        if not stack_trace:
            return None, None

        first_line = stack_trace.splitlines()[0].strip()
        if not first_line:
            return None, None

        type_candidate, _, remainder = first_line.partition(":")
        if _ and (
            "." in type_candidate
            or type_candidate.endswith("Error")
            or type_candidate.endswith("Exception")
        ):
            error_type = type_candidate.strip()
            message = remainder.strip() or None
            return error_type, message

        return None, first_line

    @staticmethod
    def _local_name(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag
