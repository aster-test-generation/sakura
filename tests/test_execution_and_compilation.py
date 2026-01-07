from __future__ import annotations

import textwrap

import pytest

from nltest.utils.compilation.maven import JavaMavenCompilation
from nltest.utils.execution.maven import JavaMavenExecution
from nltest.utils.file_io.test_file_manager import TestFileInfo, TestFileManager
from nltest.utils.pretty.prints import pretty_print


def test_get_compilation_errors_with_transient_test_file(petclinic_paths):
    compiler = JavaMavenCompilation(petclinic_paths.project_root)
    compiler.is_spring_project = True  # Skip spring-javaformat validation for generated files
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
    passing_saved = TestFileInfo(qualified_class_name=passing_name, test_code=passing_code)

    try:
        execution_issues = executor.get_execution_errors(qualified_class_name=passing_name)
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
    failing_saved = TestFileInfo(qualified_class_name=failing_name, test_code=failing_code)

    try:
        issues_with_failure = executor.get_execution_errors(qualified_class_name=failing_name)
        pretty_print("java execution errors", issues_with_failure)
        assert issues_with_failure
    finally:
        manager.delete_single(failing_saved, encode_class_name=False, strict=False)

def test_compilation_of_specific(petclinic_paths, petclinic_analysis):
    compiler = JavaMavenCompilation(petclinic_paths.project_root)
    compiler.is_spring_project = True  # Skip spring-javaformat validation for generated files

    initial_errors = compiler.get_compilation_errors()
    assert initial_errors == []

    analysis = petclinic_analysis
    class_under_test = "org.springframework.samples.petclinic.owner.PetControllerUpdateTest"
    method_under_test = "testUpdatePetForm()"

    print(analysis.get_class(class_under_test))
    print(analysis.get_method(class_under_test, method_under_test))

    assert analysis.get_class(class_under_test) is not None
    assert analysis.get_method(class_under_test, method_under_test) is not None
