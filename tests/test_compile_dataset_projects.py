from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from sakura.utils.compilation.maven import CompilationError
from scripts.utilities import compile_dataset_projects


def mock_main_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    project_dirs: list[Path],
    compile_project: MagicMock,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    reset_submodules = MagicMock()
    iter_project_dirs = MagicMock(return_value=project_dirs)
    log_summary = MagicMock()
    monkeypatch.setattr(
        compile_dataset_projects.GitUtilities,
        "reset_submodules_in_dir",
        reset_submodules,
    )
    monkeypatch.setattr(
        compile_dataset_projects, "iter_project_dirs", iter_project_dirs
    )
    monkeypatch.setattr(compile_dataset_projects, "compile_project", compile_project)
    monkeypatch.setattr(compile_dataset_projects, "log_summary", log_summary)
    monkeypatch.setattr(compile_dataset_projects.RichLog, "info", MagicMock())
    monkeypatch.setattr(compile_dataset_projects.RichLog, "error", MagicMock())
    return reset_submodules, iter_project_dirs, log_summary


def test_main_returns_zero_when_all_projects_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dirs = [Path("/datasets/alpha"), Path("/datasets/beta")]
    compile_project = MagicMock(return_value=[])
    reset_submodules, iter_project_dirs, log_summary = mock_main_dependencies(
        monkeypatch, project_dirs, compile_project
    )

    exit_status = compile_dataset_projects.main()

    assert exit_status == 0
    reset_submodules.assert_called_once_with(compile_dataset_projects.PROJECTS_DIR)
    iter_project_dirs.assert_called_once_with(compile_dataset_projects.PROJECTS_DIR)
    compile_project.assert_has_calls([call(project_dirs[0]), call(project_dirs[1])])
    log_summary.assert_called_once_with(["alpha", "beta"], [])


def test_main_returns_one_when_project_has_compilation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = Path("/datasets/broken")
    compilation_error = CompilationError(
        file="Broken.java",
        line=1,
        message="cannot find symbol",
    )
    compile_project = MagicMock(return_value=[compilation_error])
    reset_submodules, iter_project_dirs, log_summary = mock_main_dependencies(
        monkeypatch, [project_dir], compile_project
    )
    log_compilation_errors = MagicMock()
    monkeypatch.setattr(
        compile_dataset_projects,
        "log_compilation_errors",
        log_compilation_errors,
    )

    exit_status = compile_dataset_projects.main()

    assert exit_status == 1
    reset_submodules.assert_called_once_with(compile_dataset_projects.PROJECTS_DIR)
    iter_project_dirs.assert_called_once_with(compile_dataset_projects.PROJECTS_DIR)
    compile_project.assert_called_once_with(project_dir)
    log_compilation_errors.assert_called_once_with("broken", [compilation_error])
    log_summary.assert_called_once_with([], ["broken"])


def test_main_returns_one_when_project_compilation_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = Path("/datasets/exceptional")
    compile_project = MagicMock(side_effect=RuntimeError("compiler crashed"))
    reset_submodules, iter_project_dirs, log_summary = mock_main_dependencies(
        monkeypatch, [project_dir], compile_project
    )

    exit_status = compile_dataset_projects.main()

    assert exit_status == 1
    reset_submodules.assert_called_once_with(compile_dataset_projects.PROJECTS_DIR)
    iter_project_dirs.assert_called_once_with(compile_dataset_projects.PROJECTS_DIR)
    compile_project.assert_called_once_with(project_dir)
    log_summary.assert_called_once_with([], ["exceptional"])
