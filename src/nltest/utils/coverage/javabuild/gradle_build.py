"""
Gradle Build Class
"""

import os
import subprocess
import sys
from typing import List, Dict

from nltest.utils.constants import MutationOperators
from nltest.utils.coverage.javabuild import AbstractBuild
from nltest.utils.pretty import RichLog


class GradleBuild(AbstractBuild):
    """This class performs all Gradle build tasks"""

    BUILD_CMD = "gradle.cmd" if sys.platform == "win32" else "gradle"

    dependencies_init_script_file = "wca-deps-init.gradle"

    def __init__(
        self,
        project_root: str,
        build_file_name: str = "build.gradle",
        options: list = None,
        target_module: str = None,
    ) -> None:
        super().__init__(
            project_root=project_root,
            build_file_name=build_file_name,
            options=options,
            target_module=target_module,
        )
        self.build_system = "Gradle"

    def get_modules(self) -> Dict[str, Dict]:
        """Returns the hierarchy of modules for a multi-module project"""
        raise NotImplementedError

    def get_java_version(self) -> str:
        """Returns the java version by looking in the build file"""
        raise NotImplementedError

    def run_tests(self, target_tests: str = None, build_file: str = None) -> str:
        """This method calls maven to run the test class

        Args:
            target_tests (str): Pattern specifying tests to be run
            build_file (str): Use the specified build file instead of the default build file

        Returns:
            str: The output of test execution
        """
        # run mvn test for a specific class and method
        command = [
            self.BUILD_CMD,
            "--build-file",
            build_file if build_file else os.path.join(self.project_root, self.build_file_name),
            "--init-script",
            os.path.join(self.project_root, self.dependencies_init_script_file),
            "test",
            (["--tests", target_tests] if target_tests else [])
        ]
        # run command and return the output
        try:
            output = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            return output.stdout
        except subprocess.CalledProcessError as e:
            RichLog.error(f'Error running command "{e.cmd}": {e.stderr}')
            return ""

    def add_code_coverage_dependencies(self, output_build_file: str, add_java_agent: bool = False) -> None:
        """Injects build dependencies for the coverage frameworks"""
        raise NotImplementedError

    def add_mutation_analysis_dependencies(self, output_build_file: str,
                                           target_tests: List[str],
                                           excluded_tests: List[str],
                                           mutation_operators: MutationOperators = MutationOperators.defaults):
        """Injects build dependencies for mutation analysis"""
        raise NotImplementedError
