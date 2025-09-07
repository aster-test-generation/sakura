"""
Abstract Build Class
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict

from nltest.utils.constants import MutationOperators


class AbstractBuild(ABC):
    """This is the Abstract Base Class for all Builders"""

    def __init__(
        self, project_root: str, build_file_name: str = None, options: list = None,
            target_module: str = None
    ) -> None:
        self.build_system = "Abstract"
        self.project_root = Path(project_root)
        self.build_file_name = build_file_name
        self.build_file = self.project_root.joinpath(build_file_name)
        self.options = options if options else []
        self.target_module = target_module
        super().__init__()

    @abstractmethod
    def get_modules(self) -> Dict[str, Dict]:
        """Returns the hierarchy of modules for a multi-module project"""

    @abstractmethod
    def get_java_version(self) -> str:
        """Returns the java version by looking in the build file"""

    @abstractmethod
    def run_tests(self, target_tests: str = None, build_file: str = None) -> str:
        """Runs the tests using the implemented builder"""

    @abstractmethod
    def add_code_coverage_dependencies(self, output_build_file: str, add_java_agent: bool = False):
        """Injects build dependencies for the coverage frameworks"""

    @abstractmethod
    def add_mutation_analysis_dependencies(self, output_build_file: str,
                                           target_tests: List[str],
                                           excluded_tests: List[str],
                                           mutation_operators: MutationOperators = MutationOperators.defaults):
        """Injects build dependencies for mutation analysis"""
