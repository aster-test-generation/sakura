from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from sakura.utils.coverage.individual_test_coverage import (
    TEST_WATCHER_CLASS_NAME,
    IndividualTestCoverage,
    _build_jacoco_report_command,
)


class _FakeBuilder:
    def __init__(self, failure_phase: str | None = None) -> None:
        self.failure_phase = failure_phase
        self.build_files: list[Path] = []
        self.run_build_files: list[Path] = []

    def add_code_coverage_dependencies(
        self, output_build_file: str, add_java_agent: bool = False
    ) -> None:
        build_file = Path(output_build_file)
        self.build_files.append(build_file)
        build_file.write_bytes(b"temporary coverage pom")
        if self.failure_phase == "setup":
            raise RuntimeError("setup failed")

    def run_tests(
        self, target_tests: str | None = None, build_file: str | None = None
    ) -> str:
        assert build_file is not None
        self.run_build_files.append(Path(build_file))
        if self.failure_phase == "run":
            raise RuntimeError("run failed")
        return ""


def _create_coverage_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure_phase: str | None = None,
    test_names: tuple[str, ...] = ("ExampleTest",),
) -> tuple[IndividualTestCoverage, _FakeBuilder, dict[str, Path]]:
    project_root = tmp_path.joinpath("project")
    source_dir = project_root.joinpath("src/main/java/com/example")
    test_dir = project_root.joinpath("src/test/java/com/example")
    source_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    source_dir.joinpath("Example.java").write_text(
        "package com.example; public class Example {}\n", encoding="utf-8"
    )

    test_paths: dict[str, Path] = {}
    for test_name in test_names:
        test_path = test_dir.joinpath(f"{test_name}.java")
        test_path.write_text(
            f"package com.example;\npublic class {test_name} {{}}\n",
            encoding="utf-8",
        )
        test_paths[test_name] = test_path

    builder = _FakeBuilder(failure_phase)
    monkeypatch.setattr(
        "sakura.utils.coverage.individual_test_coverage.BuildFactory.create",
        lambda *_args, **_kwargs: builder,
    )
    runner = IndividualTestCoverage(project_root=project_root)
    return runner, builder, test_paths


def _watcher_path(runner: IndividualTestCoverage) -> Path:
    return runner.project_root.joinpath(
        runner.test_root,
        "com/example",
        f"{TEST_WATCHER_CLASS_NAME}.java",
    )


def _stub_collection(
    failure_phase: str | None,
):
    def collect(
        _runner: IndividualTestCoverage, _executed_tests: list[tuple[str, str]]
    ) -> tuple[dict[str, list[dict[str, object]]], list[str]]:
        if failure_phase == "collection":
            raise RuntimeError("collection failed")
        return {"com.example.ExampleTest": []}, []

    return collect


def test_extract_branch_lines_returns_three_empty_lists_without_source() -> None:
    soup = BeautifulSoup("<html><body>No source block</body></html>", "html.parser")

    extract_branch_lines = getattr(
        IndividualTestCoverage,
        "_IndividualTestCoverage__extract_branch_lines_from_html",
    )
    branch_lines = extract_branch_lines(soup)

    assert branch_lines == ([], [], [])


def test_report_command_keeps_nested_class_and_space_paths_as_single_args(
    tmp_path: Path,
) -> None:
    project_root = tmp_path.joinpath("project with spaces")
    command, exec_file, report_dir = _build_jacoco_report_command(
        project_root,
        "source files/main/java",
        ("com.example.Outer$NestedTest", "handlesInput(java.lang.String)"),
    )

    expected_root = project_root.absolute()
    assert exec_file == expected_root.joinpath(
        "target/jacoco-tests/com.example.Outer$NestedTest__handlesInput.exec"
    )
    assert report_dir == expected_root.joinpath(
        "target/com.example.Outer$NestedTest__handlesInput__report"
    )
    assert command[4] == str(exec_file)
    assert command[6] == str(expected_root.joinpath("target/classes"))
    assert command[8] == str(expected_root.joinpath("source files/main/java"))
    assert command[10] == str(report_dir)
    assert "\\$" not in "".join(command)


@pytest.mark.parametrize("preexisting_watcher", [False, True])
def test_generate_restores_files_and_removes_unique_pom_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preexisting_watcher: bool,
) -> None:
    runner, builder, test_paths = _create_coverage_runner(tmp_path, monkeypatch)
    test_path = test_paths["ExampleTest"]
    original_test = test_path.read_bytes()
    fixed_pom = runner.project_root.joinpath("pom_cov.xml")
    fixed_pom.write_bytes(b"pre-existing fixed pom")
    watcher_path = _watcher_path(runner)
    watcher_sentinel = b"pre-existing watcher\x00\xff"
    if preexisting_watcher:
        watcher_path.write_bytes(watcher_sentinel)
    monkeypatch.setattr(
        IndividualTestCoverage,
        "_IndividualTestCoverage__collect_coverage",
        _stub_collection(None),
    )

    coverage = runner.generate([("com.example.ExampleTest", "exampleTest")])

    assert coverage == {"com.example.ExampleTest": []}
    assert test_path.read_bytes() == original_test
    if preexisting_watcher:
        assert watcher_path.read_bytes() == watcher_sentinel
    else:
        assert not watcher_path.exists()
    assert fixed_pom.read_bytes() == b"pre-existing fixed pom"
    assert len(builder.build_files) == 1
    assert builder.run_build_files == builder.build_files
    assert builder.build_files[0] != fixed_pom
    assert builder.build_files[0].parent == runner.project_root
    assert not builder.build_files[0].exists()


@pytest.mark.parametrize("failure_phase", ["setup", "run", "collection"])
def test_generate_restores_files_after_pipeline_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    runner, builder, test_paths = _create_coverage_runner(
        tmp_path, monkeypatch, failure_phase=failure_phase
    )
    test_path = test_paths["ExampleTest"]
    original_test = test_path.read_bytes()
    fixed_pom = runner.project_root.joinpath("pom_cov.xml")
    fixed_pom.write_bytes(b"fixed pom sentinel")
    watcher_path = _watcher_path(runner)
    watcher_sentinel = b"watcher sentinel\x00\xff"
    watcher_path.write_bytes(watcher_sentinel)
    monkeypatch.setattr(
        IndividualTestCoverage,
        "_IndividualTestCoverage__collect_coverage",
        _stub_collection(failure_phase),
    )

    with pytest.raises(RuntimeError, match=f"{failure_phase} failed"):
        runner.generate([("com.example.ExampleTest", "exampleTest")])

    assert test_path.read_bytes() == original_test
    assert watcher_path.read_bytes() == watcher_sentinel
    assert fixed_pom.read_bytes() == b"fixed pom sentinel"
    assert len(builder.build_files) == 1
    assert not builder.build_files[0].exists()


def test_generate_restores_all_tests_after_partial_second_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, builder, test_paths = _create_coverage_runner(
        tmp_path,
        monkeypatch,
        test_names=("FirstTest", "SecondTest"),
    )
    originals = {name: path.read_bytes() for name, path in test_paths.items()}
    second_test = test_paths["SecondTest"]
    original_write_text = Path.write_text
    write_failed = False

    def fail_second_write(
        path: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        nonlocal write_failed
        if path == second_test and not write_failed:
            write_failed = True
            original_write_text(path, data[:10], encoding="utf-8")
            raise RuntimeError("second write failed")
        return original_write_text(
            path,
            data,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "write_text", fail_second_write)

    with pytest.raises(RuntimeError, match="second write failed"):
        runner.generate(
            [
                ("com.example.FirstTest", "firstTest"),
                ("com.example.SecondTest", "secondTest"),
            ]
        )

    assert {name: path.read_bytes() for name, path in test_paths.items()} == originals
    assert not _watcher_path(runner).exists()
    assert len(builder.build_files) == 1
    assert not builder.build_files[0].exists()


def test_generate_removes_temp_pom_when_cleanup_itself_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, builder, _ = _create_coverage_runner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        IndividualTestCoverage,
        "_IndividualTestCoverage__collect_coverage",
        _stub_collection(None),
    )
    monkeypatch.setattr(
        IndividualTestCoverage,
        "_IndividualTestCoverage__cleanup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )

    with pytest.raises(ExceptionGroup, match="Individual test coverage cleanup failed"):
        runner.generate([("com.example.ExampleTest", "exampleTest")])

    assert len(builder.build_files) == 1
    assert not builder.build_files[0].exists()
