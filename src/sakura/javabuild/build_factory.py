"""
Build Factory Class

Vendored from aster-test-generation/javabuild
(branch feat/maven-module-scoped-runs, commit 5ee0c60). Only the Maven
builder was vendored; sakura never uses Gradle, so GradleBuild was left
behind in the upstream repo.
"""
from sakura.javabuild.abstract_build import AbstractBuild
from sakura.javabuild.maven_build import MavenBuild


# pylint: disable=too-few-public-methods
class BuildFactory:
    """This is a factory class that produces subclasses of AbstractBuild"""

    @staticmethod
    def create(build_type: str, *args, **kwargs) -> AbstractBuild:
        """This is the factory method that returns a subclass of AbstractBuild
        depending on the value of build_type

        Args:
            build_type (str): The type of builder to instantiate
            project_root (str): The root of the project
            build_file_name (str): Optional name of the build file
            options (list, optional): Any optional additional options. Defaults to None.

        Returns:
            AbstractBuild: A concrete Build class
        """
        if build_type.lower() == "maven":
            return MavenBuild(*args, **kwargs)

        if build_type.lower() == "gradle":
            raise NotImplementedError(
                "GradleBuild was not vendored into sakura; see "
                "aster-test-generation/javabuild for the upstream implementation"
            )

        raise ValueError("Invalid Build Type")
