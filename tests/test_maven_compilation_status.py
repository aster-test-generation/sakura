from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolCall

import sakura.nl2test.generation.common.compilation_execution as compilation_execution
import sakura.nl2test.pipeline as pipeline_module
from sakura.nl2test.generation.common.compilation_execution import (
    CompilationExecutionMixin,
)
from sakura.nl2test.models import AgentState
from sakura.ray_utils.nl2test_actor import _project_compilation_status
from sakura.utils.compilation.maven import (
    CompilationError,
    CompilationScopeResult,
    JavaMavenCompilation,
)


class DummyCompilationExecutor(CompilationExecutionMixin):
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root


@pytest.mark.parametrize(
    ("returncode", "output"),
    [
        (-1, "Maven command timed out after 1 seconds"),
        (1, "Non-parseable POM: expected START_TAG"),
    ],
)
def test_compile_scope_preserves_unparsed_command_failure(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    output: str,
) -> None:
    compiler = object.__new__(JavaMavenCompilation)
    compiler.target_module = ""
    compile_tests = MagicMock(
        return_value=subprocess.CompletedProcess(
            args=["mvn", "test-compile"],
            returncode=returncode,
            stdout=output,
        )
    )
    monkeypatch.setattr(compiler, "compile_tests_proc", compile_tests)

    result = compiler.compile_scope(timeout=1)

    assert result.success is False
    assert result.errors == []
    assert result.output == output


def test_compile_scope_stops_after_dependency_preparation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler = object.__new__(JavaMavenCompilation)
    compiler.target_module = "service"
    dependency_output = "Could not resolve dependencies for project"
    install_dependencies = MagicMock(
        return_value=subprocess.CompletedProcess(
            args=["mvn", "install"],
            returncode=1,
            stdout=dependency_output,
        )
    )
    compile_tests = MagicMock()
    monkeypatch.setattr(
        compiler,
        "install_selected_projects_skip_tests_proc",
        install_dependencies,
    )
    monkeypatch.setattr(compiler, "compile_tests_proc", compile_tests)

    result = compiler.compile_scope()

    assert result.success is False
    assert result.errors == []
    assert result.output == dependency_output
    compile_tests.assert_not_called()


@pytest.mark.parametrize(
    "output",
    [
        "Maven command timed out after 600 seconds",
        "Could not resolve dependencies for project",
        "Non-readable POM /project/pom.xml",
    ],
)
def test_compile_and_execute_stops_on_unparsed_maven_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    output: str,
) -> None:
    compilation_result = CompilationScopeResult(
        success=False,
        output=output,
        errors=[],
    )
    compiler = MagicMock()
    compiler.compile_scope.return_value = compilation_result
    compiler_factory = MagicMock(return_value=compiler)
    execution_factory = MagicMock()
    monkeypatch.setattr(
        compilation_execution,
        "JavaMavenCompilation",
        compiler_factory,
    )
    monkeypatch.setattr(
        compilation_execution,
        "JavaMavenExecution",
        execution_factory,
    )
    executor = DummyCompilationExecutor(tmp_path)
    state = AgentState(
        package="com.example",
        class_name="GeneratedTest",
        method_signature="generatedTest()",
    )
    tool_call = cast(
        ToolCall,
        {"name": "compile_and_execute_test", "id": "call-1", "args": {}},
    )
    outputs = []

    executor.process_compile_and_execute(tool_call, state, outputs)

    payload = json.loads(outputs[0].content)
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "project_compilation_error"
    assert payload["error"]["details"]["files_with_errors"] == []
    assert payload["error"]["details"]["error_details"] == [compilation_result.output]
    execution_factory.assert_not_called()


def test_actor_baseline_marks_unparsed_command_failure() -> None:
    compilation_result = CompilationScopeResult(
        success=False,
        output="Maven command timed out after 600 seconds",
        errors=[],
    )

    compilation_failed, payload = _project_compilation_status(
        compilation_result, "demo"
    )

    assert compilation_failed is True
    assert payload is not None
    assert payload["project_compilation_failed"] is True
    assert payload["files_with_errors"] == []
    assert payload["error_details"] == [compilation_result.output]


def test_pipeline_project_compilation_preserves_legacy_error_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    compilation_error = CompilationError(
        file="Broken.java",
        line=3,
        message="cannot find symbol",
    )
    compilation_result = CompilationScopeResult(
        success=False,
        output="BUILD FAILURE",
        errors=[compilation_error],
    )
    compiler = MagicMock()
    compiler.compile_scope.return_value = compilation_result
    compiler.get_compilation_errors.return_value = [compilation_error]
    compiler_factory = MagicMock(return_value=compiler)
    monkeypatch.setattr(
        pipeline_module,
        "JavaMavenCompilation",
        compiler_factory,
    )
    pipeline = object.__new__(pipeline_module.Pipeline)
    pipeline.project_root = tmp_path

    assert pipeline.run_project_compilation() == [compilation_error]
    assert pipeline.run_project_compilation_scope() is compilation_result


@pytest.mark.parametrize(
    ("compilation_result", "expected"),
    [
        (
            CompilationScopeResult(
                success=False,
                output="Malformed POM",
                errors=[],
            ),
            False,
        ),
        (
            CompilationScopeResult(
                success=False,
                output="BUILD FAILURE",
                errors=[
                    CompilationError(
                        file="src/test/java/com/example/UnrelatedTest.java",
                        line=3,
                        message="cannot find symbol",
                    )
                ],
            ),
            True,
        ),
        (
            CompilationScopeResult(
                success=False,
                output="BUILD FAILURE",
                errors=[
                    CompilationError(
                        file="src/test/java/com/example/GeneratedTest.java",
                        line=3,
                        message="cannot find symbol",
                    )
                ],
            ),
            False,
        ),
    ],
)
def test_pipeline_target_compile_flag_uses_command_and_diagnostic_status(
    compilation_result: CompilationScopeResult,
    expected: bool,
) -> None:
    assert (
        pipeline_module._target_compiles(
            compilation_result,
            "com.example.GeneratedTest",
        )
        is expected
    )
