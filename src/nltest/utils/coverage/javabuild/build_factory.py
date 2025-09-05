"""
Build Factory Class
"""
from nltest.utils.coverage.javabuild import AbstractBuild, MavenBuild, GradleBuild


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
            return GradleBuild(*args, **kwargs)

        raise ValueError("Invalid Build Type")
