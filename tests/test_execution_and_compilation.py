from __future__ import annotations

import json
import textwrap
from typing import cast

import pytest
from langchain_core.messages import ToolCall

from sakura.nl2test.generation.common.compilation_execution import (
    CompilationExecutionMixin,
)
from sakura.nl2test.models import AgentState
from sakura.utils.compilation.maven import CompilationError, JavaMavenCompilation
from sakura.utils.execution.maven import JavaMavenExecution
from sakura.utils.file_io.test_file_manager import TestFileInfo, TestFileManager
from sakura.utils.pretty.prints import pretty_print


class DummyCompilationExecutor(CompilationExecutionMixin):
    def __init__(self, project_root):
        self.project_root = project_root


def test_get_compilation_errors_with_transient_test_file(petclinic_paths):
    compiler = JavaMavenCompilation(petclinic_paths.project_root)
    compiler.is_spring_project = (
        True  # Skip spring-javaformat validation for generated files
    )
    manager = TestFileManager(petclinic_paths.project_root)

    initial_errors = compiler.get_compilation_errors()
    assert initial_errors == []

    java_code = textwrap.dedent(
        """
        package org.springframework.samples.petclinic;

        import org.junit.jupiter.api.Test;

        public class BrokenCompilationTest {
            @Test
            void failsToCompile() {
                int notANumber = "definitely not a number";
            }
        }
        """
    ).strip()

    transient_info = TestFileInfo(
        qualified_class_name="org.springframework.samples.petclinic.BrokenCompilationTest",
        test_code=java_code,
    )
    qualified_name, _ = manager.save_single(
        transient_info,
        encode_class_name=False,
        sync_names=True,
        allow_overwrite=True,
    )
    saved_info = TestFileInfo(qualified_class_name=qualified_name, test_code=java_code)

    try:
        errors_with_bad_test = compiler.get_compilation_errors()
        pretty_print("java compilation errors", errors_with_bad_test)

        assert len(errors_with_bad_test) == 1
        assert manager.delete_single(saved_info, encode_class_name=False, strict=True)
        saved_info = None
    finally:
        if saved_info:
            manager.delete_single(saved_info, encode_class_name=False, strict=False)

    cleared_errors = compiler.get_compilation_errors()
    assert cleared_errors == []


def test_compile_and_execute_matches_target_paths(monkeypatch, petclinic_paths):
    executor = DummyCompilationExecutor(petclinic_paths.project_root)
    state = AgentState(
        package="org.springframework.samples.petclinic",
        class_name="BrokenCompilationTest",
        method_signature="failsToCompile()",
    )
    tool_call = cast(
        ToolCall,
        {"name": "compile_and_execute_test", "id": "call-1", "args": {}},
    )

    relative_error = CompilationError(
        file=(
            "src/test/java/org/springframework/samples/petclinic/"
            "BrokenCompilationTest.java"
        ),
        line=1,
        column=None,
        message="boom",
    )
    monkeypatch.setattr(
        JavaMavenCompilation,
        "get_compilation_errors",
        lambda self: [relative_error],
    )

    outputs = []
    executor.process_compile_and_execute(tool_call, state, outputs)
    payload = json.loads(outputs[0].content)
    assert payload["status"] == "ok"
    compilation = payload["data"]["compilation"]
    assert compilation["has_errors_for_target"] is True
    assert compilation["target_class_file"] == "BrokenCompilationTest.java"
    assert payload["data"]["execution"]["status"] == "compilation_errors"

    basename_error = CompilationError(
        file="BrokenCompilationTest.java",
        line=1,
        column=None,
        message="boom",
    )
    monkeypatch.setattr(
        JavaMavenCompilation,
        "get_compilation_errors",
        lambda self: [basename_error],
    )

    outputs = []
    executor.process_compile_and_execute(tool_call, state, outputs)
    payload = json.loads(outputs[0].content)
    assert payload["data"]["compilation"]["has_errors_for_target"] is True


def test_get_execution_errors_with_transient_test_file(petclinic_paths):
    executor = JavaMavenExecution(petclinic_paths.project_root)
    executor.is_spring_project = True
    manager = TestFileManager(petclinic_paths.project_root)

    qualified_class = "org.springframework.samples.petclinic.TransientExecutionTest"
    passing_code = textwrap.dedent(
        """
        package org.springframework.samples.petclinic;

        import org.junit.jupiter.api.Test;

        public class TransientExecutionTest {
            @Test
            void passes() {
                int value = 2;
                if (value != 2) {
                    throw new IllegalStateException("Should never happen");
                }
            }
        }
        """
    ).strip()

    passing_info = TestFileInfo(
        qualified_class_name=qualified_class,
        test_code=passing_code,
    )
    passing_name, _ = manager.save_single(
        passing_info,
        encode_class_name=False,
        sync_names=True,
        allow_overwrite=True,
    )
    passing_saved = TestFileInfo(
        qualified_class_name=passing_name, test_code=passing_code
    )

    try:
        execution_issues = executor.get_execution_errors(
            qualified_class_name=passing_name
        )
        assert execution_issues == []
    finally:
        manager.delete_single(passing_saved, encode_class_name=False, strict=False)

    failing_code = textwrap.dedent(
        """
        package org.springframework.samples.petclinic;

        import org.junit.jupiter.api.Test;

        public class TransientExecutionTest {
            @Test
            void failsAtRuntime() {
                Object value = null;
                value.toString();
            }
        }
        """
    ).strip()

    failing_info = TestFileInfo(
        qualified_class_name=qualified_class,
        test_code=failing_code,
    )
    failing_name, _ = manager.save_single(
        failing_info,
        encode_class_name=False,
        sync_names=True,
        allow_overwrite=True,
    )
    failing_saved = TestFileInfo(
        qualified_class_name=failing_name, test_code=failing_code
    )

    try:
        issues_with_failure = executor.get_execution_errors(
            qualified_class_name=failing_name
        )
        pretty_print("java execution errors", issues_with_failure)
        assert issues_with_failure
    finally:
        manager.delete_single(failing_saved, encode_class_name=False, strict=False)


def test_compilation_of_specific(petclinic_paths, petclinic_analysis):
    compiler = JavaMavenCompilation(petclinic_paths.project_root)
    compiler.is_spring_project = (
        True  # Skip spring-javaformat validation for generated files
    )

    initial_errors = compiler.get_compilation_errors()
    assert initial_errors == []

    analysis = petclinic_analysis
    class_under_test = (
        "org.springframework.samples.petclinic.owner.PetControllerUpdateTest"
    )
    method_under_test = "testUpdatePetForm()"

    print(analysis.get_class(class_under_test))
    print(analysis.get_method(class_under_test, method_under_test))

    assert analysis.get_class(class_under_test) is not None
    assert analysis.get_method(class_under_test, method_under_test) is not None
