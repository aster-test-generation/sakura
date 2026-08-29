from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sakura.utils.execution.maven import JavaMavenExecution


MINIMAL_POM = """\
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>example</artifactId>
  <version>1.0.0</version>
</project>
"""


@pytest.fixture
def executor(tmp_path: Path) -> JavaMavenExecution:
    tmp_path.joinpath("pom.xml").write_text(MINIMAL_POM, encoding="utf-8")
    return JavaMavenExecution(tmp_path)


def _write_report(
    tmp_path: Path,
    test_name: str,
    issue_kind: str | None = "failure",
) -> Path:
    reports_dir = tmp_path.joinpath("target", "surefire-reports")
    reports_dir.mkdir(parents=True)
    issue = (
        f'<{issue_kind} message="boom" type="AssertionError">boom</{issue_kind}>'
        if issue_kind
        else ""
    )
    reports_dir.joinpath("TEST-com.example.ExampleTest.xml").write_text(
        (
            '<testsuite name="com.example.ExampleTest">'
            f'<testcase classname="com.example.ExampleTest" name="{test_name}">'
            f"{issue}</testcase></testsuite>"
        ),
        encoding="utf-8",
    )
    return reports_dir


def _mock_maven_run(
    monkeypatch: pytest.MonkeyPatch,
    executor: JavaMavenExecution,
    returncode: int,
    stdout: str,
) -> None:
    def run_tests_proc(
        target_tests: str | None = None,
        build_file: str | None = None,
        also_make: bool = True,
        timeout: int | None = 600,
    ) -> subprocess.CompletedProcess[str]:
        del target_tests, build_file, also_make, timeout
        return subprocess.CompletedProcess(
            args=["mvn", "test"], returncode=returncode, stdout=stdout
        )

    monkeypatch.setattr(executor, "run_tests_proc", run_tests_proc)


@pytest.mark.parametrize(
    "reported_name",
    ["testFoo", "testFoo(java.lang.String)", "testFoo[1]"],
)
def test_target_method_accepts_supported_surefire_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    executor: JavaMavenExecution,
    reported_name: str,
) -> None:
    reports_dir = _write_report(tmp_path, reported_name)
    _mock_maven_run(monkeypatch, executor, returncode=0, stdout="BUILD SUCCESS")
    monkeypatch.setattr(executor, "find_surefire_reports", lambda: [str(reports_dir)])

    issues = executor.get_execution_errors(
        qualified_class_name="com.example.ExampleTest",
        method_signature="testFoo()",
    )

    assert len(issues) == 1
    assert issues[0].test_name == reported_name
    assert issues[0].error_type == "AssertionError"


def test_target_method_rejects_longer_method_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    executor: JavaMavenExecution,
) -> None:
    reports_dir = _write_report(tmp_path, "testFoobar")
    _mock_maven_run(monkeypatch, executor, returncode=0, stdout="BUILD SUCCESS")
    monkeypatch.setattr(executor, "find_surefire_reports", lambda: [str(reports_dir)])

    issues = executor.get_execution_errors(
        qualified_class_name="com.example.ExampleTest",
        method_signature="testFoo()",
    )

    assert len(issues) == 1
    assert issues[0].error_type == "NoSpecifiedTests"


@pytest.mark.parametrize(
    ("returncode", "qualified_class", "expected_type"),
    [
        (-1, "com.example.ExampleTest", "MavenTimeout"),
        (2, "com.example.ExampleTest", "MavenBuildFailure"),
        (-1, None, "MavenTimeout"),
        (2, None, "MavenBuildFailure"),
    ],
)
def test_nonzero_maven_status_is_reported_for_all_run_scopes(
    monkeypatch: pytest.MonkeyPatch,
    executor: JavaMavenExecution,
    returncode: int,
    qualified_class: str | None,
    expected_type: str,
) -> None:
    _mock_maven_run(
        monkeypatch,
        executor,
        returncode=returncode,
        stdout="dependency resolution failed",
    )
    monkeypatch.setattr(executor, "find_surefire_reports", lambda: [])

    issues = executor.get_execution_errors(
        qualified_class_name=qualified_class,
        method_signature="testFoo()" if qualified_class else None,
    )

    assert len(issues) == 1
    assert issues[0].error_type == expected_type
    assert "Output: dependency resolution failed" in (issues[0].message or "")


@pytest.mark.parametrize(
    ("returncode", "expected_type"),
    [(-1, "MavenTimeout"), (2, "MavenBuildFailure")],
)
def test_nonzero_maven_status_overrides_matching_passing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    executor: JavaMavenExecution,
    returncode: int,
    expected_type: str,
) -> None:
    reports_dir = _write_report(tmp_path, "testFoo", issue_kind=None)
    _mock_maven_run(monkeypatch, executor, returncode, "Maven failed")
    monkeypatch.setattr(executor, "find_surefire_reports", lambda: [str(reports_dir)])

    issues = executor.get_execution_errors(
        qualified_class_name="com.example.ExampleTest",
        method_signature="testFoo()",
    )

    assert len(issues) == 1
    assert issues[0].error_type == expected_type


def test_test_issue_takes_precedence_over_nonzero_maven_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    executor: JavaMavenExecution,
) -> None:
    reports_dir = _write_report(tmp_path, "testFoo")
    _mock_maven_run(monkeypatch, executor, returncode=2, stdout="BUILD FAILURE")
    monkeypatch.setattr(executor, "find_surefire_reports", lambda: [str(reports_dir)])

    issues = executor.get_execution_errors(
        qualified_class_name="com.example.ExampleTest",
        method_signature="testFoo()",
    )

    assert len(issues) == 1
    assert issues[0].error_type == "AssertionError"
