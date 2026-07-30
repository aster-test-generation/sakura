"""
Abstract Build Class

Vendored from aster-test-generation/javabuild
(branch feat/maven-module-scoped-runs, commit 5ee0c60) with imports rewired
to sakura's own constants.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

from sakura.utils.constants import MutationOperators


class AbstractBuild(ABC):
    """This is the Abstract Base Class for all Builders"""

    def __init__(
        self,
        project_root: str,
        build_file_name: str = None,
        options: list = None,
        target_module: str = None,
    ) -> None:
        self.build_system = "Abstract"
        self.project_root = Path(project_root)
        self.build_file_name = build_file_name
        self.build_file = self.project_root.joinpath(build_file_name)
        self.options = options if options else []
        self.target_module = target_module
        super().__init__()

    @abstractmethod
    def is_multi_module_project(self) -> bool:
        """Returns True if the project is multi-module"""

    @abstractmethod
    def get_modules(self) -> Dict[str, Dict]:
        """Returns the hierarchy of modules for a multi-module project"""

    @abstractmethod
    def get_parent_module_path(self) -> Path | None:
        """Returns the parent Path of the module"""

    @abstractmethod
    def get_java_version(self) -> str:
        """Returns the java version by looking in the build file"""

    @abstractmethod
    def compile_tests(self, pre_compile_build: bool = False) -> str:
        """Compiles the tests using the implemented builder"""

    @abstractmethod
    def is_compile_application(self) -> bool:
        """Compiles the application using the implemented builder"""

    @abstractmethod
    def run_tests(self, target_tests: str = None, build_file: str = None) -> str:
        """Runs the tests using the implemented builder"""

    @abstractmethod
    def add_wca_test_dependencies(
        self, output_build_file: str, is_add_spring_dependency: bool
    ):
        """Injects build dependencies for the test frameworks"""

    @abstractmethod
    def add_code_coverage_dependencies(
        self, output_build_file: str, add_java_agent: bool = False
    ):
        """Injects build dependencies for the coverage frameworks"""

    @abstractmethod
    def add_mutation_analysis_dependencies(
        self,
        output_build_file: str,
        target_tests: List[str],
        excluded_tests: List[str],
        target_classes: List[str],
        mutation_operators: MutationOperators = MutationOperators.defaults,
    ):
        """Injects build dependencies for mutation analysis"""

    @abstractmethod
    def run_sanitization_recipes(self):
        """Runs the recipes for sanitization"""

    @abstractmethod
    def find_compile_errors(
        self, class_name: str, compiler_output: str
    ) -> Tuple[List[int], Dict[int, str]]:
        """Finds compiler errors based on builder output

        Args:
            class_name (str): The nae of the class being compiled
            compiler_output (str): The output of the compile

        Returns:
            list: A list of line numbers that have errors
            dict: A dictionary with the line number as the key and the line as the value
        """

    @abstractmethod
    def remove_functions_with_errors(
        self, class_name: str, compiler_output: str, error_and_code_body: dict
    ) -> list:
        """Removes functions that contain errors

        Args:
            class_name (str): The name of the class under test
            compiler_output (str): The output of the maven compiler
            error_and_code_body (dict): A dictionary of errors

        Returns:
            list: A list of function names to be removed
        """

    @abstractmethod
    def find_runtime_error_for_method(
        self, compiler_output: str, test_class_name: str, method_name: str
    ) -> Dict[int, str]:
        """Searches compiler output for errors and returns the line number and error message

        Args:
            compiler_output (str): Output of the compile step
            test_class_name (str): Name of teh class under test
            method_name (str): Name of the method under test

        Returns:
            Dict[int, str]: The line number as the key, and error line as the value
        """

    @abstractmethod
    def find_runtime_error_for_class(
        self, compiler_output: str, test_class_name: str
    ) -> Dict[int, str]:
        """Process the runtime errors from a compile
        Args:
            compiler_output (str): Output of the compile step
            test_class_name (str): Name of test class
        Returns:
            Dict[int, str]: The line number as the key, and error line as the value
        """

    @abstractmethod
    def get_class_names_with_compile_errors(self, compiler_output: str) -> List[str]:
        """Returns a list of class names that had compiler errors

        Args:
            compiler_output (str): Output from a test compile

        Returns:
            List[str]: A list of class names that had errors when compiled
        """
