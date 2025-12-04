import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel

from javabuild.maven_build import MavenBuild

from nltest.utils.analysis import CommonAnalysis


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

    def __init__(self, project_root: Path):
        super().__init__(str(project_root))

    def get_execution_errors(
            self,
            qualified_class_name: Optional[str] = None,
            method_signature: Optional[str] = None,
    ) -> List[ExecutionIssue]:
        target_tests: Optional[str] = None

        if qualified_class_name:
            simple_class = CommonAnalysis.get_simple_class_name(qualified_class_name)
            if method_signature:
                method_name = CommonAnalysis.get_simple_method_name(method_signature)
                target_tests = f"{simple_class}#{method_name}"
            else:
                target_tests = simple_class

        run_started_at = time.time()
        runtime_output = self.run_tests(target_tests=target_tests)

        report_dirs = self.find_surefire_reports()
        if report_dirs:
            issues = self.parse_surefire_reports(report_dirs, min_mtime=run_started_at)
            if issues:
                return issues
        return self.parse_console_output(runtime_output)

    def find_surefire_reports(self) -> List[str]:
        """Locate every target/surefire-reports directory in the project tree."""

        found: List[str] = []
        for dirpath, dirnames, _ in os.walk(self.project_root):
            if "surefire-reports" in dirnames and os.path.basename(dirpath) == "target":
                found.append(os.path.join(dirpath, "surefire-reports"))
        return found

    def parse_surefire_reports(self, reports_dirs: List[str], min_mtime: Optional[float] = None) -> List[
        ExecutionIssue]:
        """Read Surefire XML reports and convert every failure/error into ExecutionIssue objects."""

        issues: List[ExecutionIssue] = []
        for rdir in reports_dirs:
            try:
                files = [f for f in os.listdir(rdir) if f.startswith("TEST-") and f.endswith(".xml")]
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
                        child for child in root if self._local_name(child.tag) == "testsuite"
                    ]
                else:
                    # Fallback: treat any <testsuite> in the document as a suite
                    suites = [
                        child for child in root.iter()
                        if self._local_name(child.tag) == "testsuite"
                    ]

                for suite in suites:
                    for tc in (child for child in suite if self._local_name(child.tag) == "testcase"):
                        class_name = tc.get("classname") or ""
                        test_name = tc.get("name") or ""

                        for kind_tag in ("failure", "error"):
                            for element in (child for child in tc if self._local_name(child.tag) == kind_tag):
                                issue = ExecutionIssue(
                                    class_name=class_name,
                                    test_name=test_name,
                                    kind="failure" if kind_tag == "failure" else "error",
                                    message=element.get("message"),
                                    error_type=element.get("type"),
                                    stack_trace=self._truncate_stack_trace((element.text or "").strip() or None),
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
                    class_name = (header.group("class") or header.group("class2")).strip()
                    test_name = (header.group("test") or header.group("test2")).strip()
                    kind = "failure" if header.group("kind").upper() == "FAILURE" else "error"
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
                error_type, derived_message = self._extract_error_type_and_message(stack_trace)
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
    def _collect_stack_trace(cls, lines: List[str], start_index: int) -> Tuple[Optional[str], int]:
        collected: List[str] = []
        idx: int = start_index
        blank_streak = 0

        while idx < len(lines):
            current: str = lines[idx]
            if cls.FAILURE_HEADER_RE.match(current) or cls.FAILURE_SUMMARY_RE.match(current):
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
                entry = current[len("[ERROR]"):].lstrip()
            elif current.startswith("[INFO]"):
                entry = current[len("[INFO]"):].lstrip()

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
        head = lines[:cls.HEAD_STACKTRACE_LINES]
        tail = lines[-cls.TAIL_STACKTRACE_LINES:]

        return "\n".join(head) + f"\n    ... (skipped {skipped} lines) ...\n" + "\n".join(tail)

    @staticmethod
    def _extract_error_type_and_message(stack_trace: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if not stack_trace:
            return None, None

        first_line = stack_trace.splitlines()[0].strip()
        if not first_line:
            return None, None

        type_candidate, _, remainder = first_line.partition(":")
        if _ and ("." in type_candidate or type_candidate.endswith("Error") or type_candidate.endswith("Exception")):
            error_type = type_candidate.strip()
            message = remainder.strip() or None
            return error_type, message

        return None, first_line

    @staticmethod
    def _local_name(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag
