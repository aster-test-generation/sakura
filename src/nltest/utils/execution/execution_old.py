import os
import re
import sys
import shutil
import subprocess
import xml.etree.ElementTree as ET
from typing import Optional, List

from pydantic import BaseModel, Field

from nltest.utils.constants import MAVEN_CMD


MAX_STACKTRACE_LINES_ON_FAILURE = 12


class TestIssue(BaseModel):
    """Single failure or error in a test case."""

    class_name: str
    test_name: str
    kind: str  # "failure" or "error"
    message: Optional[str] = None
    error_type: Optional[str] = None
    time_sec: Optional[float] = None
    file: Optional[str] = None
    line: Optional[int] = None
    stack_trace: Optional[str] = None


class TestExecutionFeedback(BaseModel):
    """Structured result of a Maven/Surefire test run, with minimized logs."""

    command: str
    returncode: int
    test_class_name: Optional[str] = None
    timed_out: bool = False

    # Aggregated counts
    passed: bool
    tests_run: int
    failures: int
    errors: int
    skipped: int
    time_elapsed_sec: float

    # Details for failures/errors
    issues: List[TestIssue] = Field(default_factory=list)

    # Outputs
    stdout_summary: Optional[str] = None
    stderr: Optional[str] = None

    reports_dirs: List[str] = Field(default_factory=list)

    failure_reason_code: Optional[str] = (
        None  # e.g., TEST_FAILURES, TEST_ERRORS, NO_TESTS_MATCHED, SUREFIRE_PLUGIN_ERROR, BUILD_ERROR, TIMEOUT
    )
    failure_reason: Optional[str] = None  # human-readable one-liner


class JavaExecution:
    """
    Execute tests in a Maven-based Java project and return parsed feedback.
    Assumes projects are Maven and pre-compiled; runs only the test phase.
    """

    @staticmethod
    def execute(
        project_root: str,
        test_class_name: Optional[str],
        timeout: int = 900,
    ) -> TestExecutionFeedback:
        if not test_class_name:
            raise ValueError("test_class_name is required for this executor.")

        pom = os.path.join(project_root, "pom.xml")

        def _resolve_maven_cmd_parts(project_root: str) -> List[str]:
            wrapper = "mvnw.cmd" if sys.platform == "win32" else "mvnw"
            wrapper_path = os.path.join(project_root, wrapper)
            if os.path.isfile(wrapper_path):
                if sys.platform != "win32" and not os.access(wrapper_path, os.X_OK):
                    return ["sh", wrapper_path]
                return [wrapper_path]
            mvn_path = shutil.which(MAVEN_CMD)
            if mvn_path:
                return [mvn_path]
            raise FileNotFoundError(
                f"Maven not found. Neither '{MAVEN_CMD}' on PATH nor wrapper '{wrapper}' at {project_root}."
            )

        cmd_parts = _resolve_maven_cmd_parts(project_root) + [
            "-f",
            pom,
            "-B",  # batch mode: cleaner logs
            "-ntp",  # no transfer progress
            "-DfailIfNoTests=false",  # ignore empty modules
            f"-Dtest={test_class_name}",
            "test",
        ]

        # Run Maven
        try:
            p = subprocess.Popen(
                cmd_parts,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,  # capture strings
                universal_newlines=True,
            )
        except FileNotFoundError as e:
            return TestExecutionFeedback(
                command=" ".join(cmd_parts),
                returncode=-1,
                test_class_name=test_class_name,
                timed_out=False,
                passed=False,
                tests_run=0,
                failures=0,
                errors=0,
                skipped=0,
                time_elapsed_sec=0.0,
                issues=[],
                stdout_summary=str(e),
                stderr=str(e),
                reports_dirs=[],
                failure_reason_code="BUILD_ERROR",
                failure_reason="Maven command not found",
            )

        try:
            stdout, stderr = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            stdout, stderr = p.communicate()
            return TestExecutionFeedback(
                command=" ".join(cmd_parts),
                returncode=-1,
                test_class_name=test_class_name,
                timed_out=True,
                passed=False,
                tests_run=0,
                failures=0,
                errors=0,
                skipped=0,
                time_elapsed_sec=0.0,
                issues=[],
                stdout_summary=_minimal_console_extract(stdout),
                stderr="Test execution timed out.",
                reports_dirs=[],
                failure_reason_code="TIMEOUT",
                failure_reason="Test execution timed out",
            )

        # Parse Surefire XML reports first (most reliable)
        reports_dirs = _find_surefire_reports(project_root)
        agg = _parse_surefire_reports(reports_dirs)

        # If we still don't have numbers, fall back to console parsing
        if agg.tests_run == 0 and (agg.failures == 0 and agg.errors == 0):
            csum = _parse_console_summary(stdout)
            if csum:
                agg = csum

        # Decide pass/fail
        passed = (
            p.returncode == 0
            and agg.tests_run >= 0
            and agg.failures == 0
            and agg.errors == 0
        )

        failure_reason_code = None
        failure_reason = None
        if not passed:
            if agg.failures or agg.errors:
                failure_reason_code = "TEST_ERRORS" if agg.errors else "TEST_FAILURES"
                failure_reason = f"{agg.failures} failure(s), {agg.errors} error(s) across {agg.tests_run} test(s)"
            elif p.returncode != 0:
                failure_reason_code, failure_reason = _detect_special_failure(
                    stdout, stderr
                )

        feedback = TestExecutionFeedback(
            command=" ".join(cmd_parts),
            returncode=p.returncode,
            test_class_name=test_class_name,
            timed_out=False,
            passed=passed,
            tests_run=agg.tests_run,
            failures=agg.failures,
            errors=agg.errors,
            skipped=agg.skipped,
            time_elapsed_sec=agg.time_elapsed_sec,
            issues=agg.issues,
            stdout_summary=None if passed else _minimal_console_extract(stdout),
            stderr=(stderr or None) if not passed else None,
            reports_dirs=reports_dirs,
            failure_reason_code=failure_reason_code,
            failure_reason=failure_reason,
        )
        return feedback


class _Agg:
    """Internal aggregation structure."""

    def __init__(self):
        self.tests_run = 0
        self.failures = 0
        self.errors = 0
        self.skipped = 0
        self.time_elapsed_sec = 0.0
        self.issues: List[TestIssue] = []


def _find_surefire_reports(project_root: str) -> List[str]:
    """Find all target/surefire-reports directories (handles multi-module builds)."""
    found: List[str] = []
    for dirpath, dirnames, _ in os.walk(project_root):
        if "surefire-reports" in dirnames and os.path.basename(dirpath) == "target":
            found.append(os.path.join(dirpath, "surefire-reports"))
    return found


def _parse_surefire_reports(reports_dirs: List[str]) -> _Agg:
    agg = _Agg()
    for rdir in reports_dirs:
        try:
            files = [
                f
                for f in os.listdir(rdir)
                if f.startswith("TEST-") and f.endswith(".xml")
            ]
        except FileNotFoundError:
            continue

        for fname in files:
            fpath = os.path.join(rdir, fname)
            try:
                tree = ET.parse(fpath)
                root = tree.getroot()
            except Exception:
                continue

            # Handle <testsuite> or <testsuites>
            suites = []
            if root.tag == "testsuite":
                suites = [root]
            elif root.tag == "testsuites":
                suites = list(root.findall("testsuite"))

            for suite in suites:
                agg.tests_run += _as_int(suite.get("tests"))
                agg.failures += _as_int(suite.get("failures"))
                agg.errors += _as_int(suite.get("errors"))
                agg.skipped += _as_int(suite.get("skipped"))
                agg.time_elapsed_sec += _as_float(suite.get("time"))

                for tc in suite.findall("testcase"):
                    t_class = tc.get("classname") or ""
                    t_name = tc.get("name") or ""
                    t_time = _as_float(tc.get("time"))

                    for kind_tag in ("failure", "error"):
                        for el in tc.findall(kind_tag):
                            issue = TestIssue(
                                class_name=t_class,
                                test_name=t_name,
                                kind="failure" if kind_tag == "failure" else "error",
                                message=el.get("message"),
                                error_type=el.get("type"),
                                time_sec=t_time,
                                stack_trace=_truncate_stack_trace(
                                    (el.text or "").strip() or None
                                ),
                            )
                            _maybe_fill_file_line(issue)
                            agg.issues.append(issue)

    return agg


def _maybe_fill_file_line(issue: TestIssue) -> None:
    """Attempt to pull file and line from a Java stack trace fragment."""
    if not issue.stack_trace:
        return
    # Match "...(MyClass.java:123)" or "...(src/.../MyClass.java:123)"
    m = re.search(r"\(([^()/:\\\s]+\.java):(\d+)\)", issue.stack_trace)
    if not m:
        m = re.search(r"([A-Za-z0-9_\-./\\]+\.java):(\d+)", issue.stack_trace)
    if m:
        issue.file = m.group(1)
        try:
            issue.line = int(m.group(2))
        except ValueError:
            pass


_SUMMARY_RE = re.compile(
    r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)",
    re.IGNORECASE,
)


def _parse_console_summary(stdout: str) -> Optional[_Agg]:
    """
    Fallback when XML is missing: parse the last 'Tests run: ..., Failures: ...' summary
    from Maven/Surefire console output.
    """
    matches = list(_SUMMARY_RE.finditer(stdout or ""))
    if not matches:
        return None

    last = matches[-1]
    agg = _Agg()
    agg.tests_run = int(last.group(1))
    agg.failures = int(last.group(2))
    agg.errors = int(last.group(3))
    agg.skipped = int(last.group(4))

    # Attempt to infer elapsed time near the summary
    time_re = re.compile(r"Time elapsed:\s*([0-9]*\.?[0-9]+)\s*s", re.IGNORECASE)
    lines = (stdout or "").splitlines()
    idx = None
    for i, line in enumerate(lines):
        if last.group(0) in line:
            idx = i
    if idx is not None:
        window = "\n".join(lines[max(0, idx - 20) : idx + 20])
        tm = time_re.search(window)
        if tm:
            try:
                agg.time_elapsed_sec = float(tm.group(1))
            except ValueError:
                pass
    return agg


def _minimal_console_extract(stdout: Optional[str]) -> Optional[str]:
    """
    Return the smallest useful slice from stdout:
      * Last 'Tests run: ...' summary
      * 'Failures:' / 'Errors:' listing lines
      * 'There are test failures.' / 'BUILD FAILURE'
      * 'No tests found...' / 'No tests were executed!'
      * surefire plugin error lines
    """
    if not stdout:
        return None

    lines = (stdout or "").splitlines()

    # Try to locate Results: block near the end
    start_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if "Results:" in lines[i]:
            start_idx = i
            break

    candidates: List[str] = []

    def keep(line: str) -> bool:
        if _SUMMARY_RE.search(line):
            return True
        if line.startswith("[ERROR]"):
            # Keep only concise, useful error lines
            key_patterns = (
                "Failures:",
                "Errors:",
                "There are test failures.",
                "BUILD FAILURE",
                "No tests found",
                "No tests were executed",
                "Failed to execute goal",
                "SurefireBooterForkException",
            )
            return any(pat in line for pat in key_patterns)
        # Some summaries are INFO-level
        key_infos = (
            "Failures:",
            "Errors:",
            "There are test failures.",
            "No tests found",
            "No tests were executed",
        )
        return any(line.strip().startswith(f"[INFO] {k}") for k in key_infos)

    # Prefer the Results block window
    if start_idx is not None:
        window = lines[max(0, start_idx - 15) : min(len(lines), start_idx + 80)]
        candidates = [ln for ln in window if keep(ln)]

    # Fallback: scan last 200 lines
    if not candidates:
        tail = lines[-200:]
        candidates = [ln for ln in tail if keep(ln)]

    # Deduplicate while preserving order
    seen = set()
    filtered = []
    for ln in candidates:
        if ln not in seen:
            filtered.append(ln)
            seen.add(ln)

    return "\n".join(filtered) if filtered else None


def _detect_special_failure(stdout: Optional[str], stderr: Optional[str]) -> (str, str):
    s = f"{stdout or ''}\n{stderr or ''}"

    patterns = [
        (
            r"No tests (?:found|were executed)[:!]",
            "NO_TESTS_MATCHED",
            "No tests matched the -Dtest filter",
        ),
        (
            r"Failed to execute goal .* surefire.*:",
            "SUREFIRE_PLUGIN_ERROR",
            "Maven Surefire plugin failed",
        ),
        (
            r"SurefireBooterForkException",
            "SUREFIRE_PLUGIN_ERROR",
            "Surefire forked JVM failed",
        ),
        (r"BUILD FAILURE", "BUILD_ERROR", "Build failed"),
    ]
    for pat, code, msg in patterns:
        if re.search(pat, s, re.IGNORECASE):
            return code, msg

    return "BUILD_ERROR", "Build failed (see minimal console extract)"


def _truncate_stack_trace(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    lines = text.splitlines()
    if len(lines) <= MAX_STACKTRACE_LINES_ON_FAILURE:
        return text
    return "\n".join(lines[:MAX_STACKTRACE_LINES_ON_FAILURE]) + "\n... (truncated)"


def _as_int(val: Optional[str]) -> int:
    try:
        return int(val) if val is not None else 0
    except ValueError:
        return 0


def _as_float(val: Optional[str]) -> float:
    try:
        return float(val) if val is not None else 0.0
    except ValueError:
        return 0.0


if __name__ == "__main__":
    # Example usage (will run Maven in the given project root):
    project_root = "tests/resources/spring-petclinic/"
    execution_feedback = JavaExecution.execute(
        project_root,
        test_class_name="org.springframework.samples.petclinic.owner.OwnerControllerTests",
    )
    print()
    print()
    print(execution_feedback)
