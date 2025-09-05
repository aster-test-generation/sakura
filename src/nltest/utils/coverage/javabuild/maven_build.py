"""
Maven Build Class
"""

import subprocess
import sys
import xml.etree.ElementTree as elemtree
from pathlib import Path
from typing import List, Dict
from xml.etree.ElementTree import ElementTree

from nltest.utils import constants
from nltest.utils.constants import MutationOperators
from nltest.utils.coverage.javabuild import AbstractBuild
from nltest.utils.pretty import RichLog


class MavenBuild(AbstractBuild):
    """This class performs all Maven build tasks"""

    MAVEN_CMD = "mvn.cmd" if sys.platform == "win32" else "mvn"

    # default maven options for test execution
    DEFAULT_MAVEN_OPTIONS = [
        "-Drat.skip=true",
        "-Dlicense.skip=true",
        "-Dmaven.test.skip=false",
        "-Dmaven.test.failure.ignore=true",
    ]

    PITEST_MAVEN_VERSION = "1.19.4"
    PITEST_JUNIT5_PLUGIN_VERSION = "1.2.3"

    def __init__(
            self, project_root: str, build_file_name: str = "pom.xml",
            options: list = None, target_module: str = None
    ) -> None:
        if options:
            self.options = options
        else:
            self.options = self.DEFAULT_MAVEN_OPTIONS
        super().__init__(
            project_root=project_root,
            build_file_name=build_file_name,
            options=self.options,
            target_module=target_module,
        )
        self.build_system = "Maven"
        self.build_tree, self.build_namespaces = self.__parse_build_file()

    def __parse_build_file(self, build_file: Path = None) -> tuple[elemtree.ElementTree, dict]:
        """
        Parses the Maven build file and returns the parsed tree and the namespace info.

        Args:
            build_file (Path): Maven build file to be parsed

        Returns:
            parsed tree and namespace

        """
        tree = (
            elemtree.parse(self.build_file)
            if build_file is None
            else elemtree.parse(build_file)
        )
        root = tree.getroot()
        namespace = (
            root.tag.split("}")[0].strip("{") if root.tag.startswith("{") else None
        )
        if namespace:
            elemtree.register_namespace("", namespace)
        namespaces = {"": namespace} if namespace else None
        return tree, namespaces

    @staticmethod
    def __create_dependency_element(group_id, dependency: dict) -> elemtree.Element:
        """
        Creates and returns a <dependency> element for the given dependency.

        Args:
            dependency:

        Returns:

        """
        dependency_str = f"""
    <dependency>
      <groupId>{group_id}</groupId>
      <artifactId>{dependency["artifact_id"]}</artifactId>
      <version>{dependency["version"]}</version>
      <scope>{dependency["scope"]}</scope>
    </dependency>
"""
        return elemtree.fromstring(dependency_str)

    def __get_module_hierarchy(self, pom_path: Path, module_hierarchy: Dict) -> None:
        """
        Identifies modules recursively starting with the specified pom and adds the module
        hierarchy to the given dict.

        Args:
            pom_path: POM to perform module extraction on
            module_hierarchy: Information about extracted module hierarchy
        """
        tree, namespace = self.__parse_build_file(build_file=pom_path)
        modules = self.__get_module_names(tree=tree, namespace=namespace)
        for module in modules:
            module_hierarchy[module] = {}
            module_pom_path = pom_path.parent.joinpath(module, "pom.xml")
            if module_pom_path.exists():
                self.__get_module_hierarchy(
                    pom_path=module_pom_path,
                    module_hierarchy=module_hierarchy[module]
                )

    @staticmethod
    def __get_module_names(tree: ElementTree, namespace: Dict) -> List[str]:
        """
        Computes module names in the given XML tree (for a POM) using the give.

        Args:
            tree: XML tree for POM
            namespace: Namespace dictionary for xpath queries

        Returns:
            List[str]: List of module names
        """
        modules_element = tree.getroot().find("./modules", namespace)
        if modules_element is None:
            return []
        return [
            module.text.strip()
            for module in modules_element if module.text and module.text.strip()
        ]

    def get_modules(self) -> Dict[str, Dict]:
        """
        Returns the hierarchy of modules for a multi-module project or empty dict for a project with
        no modules.

        Returns:
            Dict: Dictionary containing module hierarchy
        """
        module_hierarchy = dict()
        self.__get_module_hierarchy(pom_path=self.build_file, module_hierarchy=module_hierarchy)
        return module_hierarchy

    def get_java_version(self) -> str:
        """Returns the java version by looking in the build file"""
        raise NotImplementedError

    def run_tests(self, target_tests: str = None, build_file: str = None) -> str:
        """This method calls maven to run the test class

        Args:
            target_tests (str): Pattern specifying tests to be run
            build_file (str): Use the specified build file instead of the default build file

        Returns:
            str: The output of test execution delimited by new lines
        """
        # run mvn test for a specific class and method
        command = [
            self.MAVEN_CMD,
            "-f",
            build_file if build_file else str(self.build_file),
            *(self.options),
            *(["-Dtest=" + target_tests] if target_tests else []),
            "test",
            "-Dstyle.color=never",
            "-Dspring-javaformat:apply"
        ]
        # if target module is specified (for a multi-module project), compile the specified module
        # and its dependencies
        if self.target_module:
            command.extend([
                "--projects",
                self.target_module,
                "--also-make"
            ])

        # run command and return the output
        RichLog.debug(f"Running command: {command}")
        output = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        return output.stdout

    def add_code_coverage_dependencies(self, output_build_file: str, add_java_agent: bool = False):
        """
        Augments a given Maven build file with Jacoco dependencies and configuration for
        collecting code coverage.

        Args:
            output_build_file (str): Path of output build file
            add_java_agent (bool): Add java agent as arg-line option for maven surefire plugin

        Returns: Path to the augmented build file

        """
        RichLog.info(f"Updating {self.project_root} build file: {self.build_file}")
        root = self.build_tree.getroot()

        # locate or create <build>/<plugins>
        build = root.find("build", self.build_namespaces)
        if build is None:
            build = elemtree.SubElement(root, "build")
        plugins = build.find("plugins", self.build_namespaces)
        if plugins is None:
            plugins = elemtree.SubElement(build, "plugins")

        # add and configure jacoco plugin if it does not exist
        if not self.__element_exists(parent_element=plugins, target_element="plugin",
                                     target_group_id="org.jacoco", target_artifact_id="jacoco-maven-plugin"):
            jacoco_plugin = elemtree.SubElement(plugins, "plugin")
            elemtree.SubElement(jacoco_plugin, "groupId").text = "org.jacoco"
            elemtree.SubElement(jacoco_plugin, "artifactId").text = "jacoco-maven-plugin"
            elemtree.SubElement(jacoco_plugin, "version").text = constants.JACOCO_VERSION

            # configure plugin
            config = elemtree.SubElement(jacoco_plugin, "configuration")
            elemtree.SubElement(config, "destFile").text = constants.MAVEN_JACOCO_COV_FILE
            elemtree.SubElement(config, "dataFile").text = constants.MAVEN_JACOCO_COV_FILE

            # add executions element
            executions = elemtree.SubElement(jacoco_plugin, "executions")
            execution = elemtree.SubElement(executions, "execution")
            goals = elemtree.SubElement(execution, "goals")
            elemtree.SubElement(goals, "goal").text = "prepare-agent"

        if add_java_agent:
            # locate or create <dependencies>
            dependencies = root.find("dependencies", self.build_namespaces)
            if dependencies is None:
                dependencies = elemtree.SubElement(root, "dependencies")

            # if jacoco agent dependency does not exist, add it
            if not self.__element_exists(parent_element=dependencies, target_element="dependency",
                                         target_group_id="org.jacoco", target_artifact_id="org.jacoco.agent"):
                dependency_elem = elemtree.SubElement(dependencies, "dependency")
                elemtree.SubElement(dependency_elem, "groupId").text = "org.jacoco"
                elemtree.SubElement(dependency_elem, "artifactId").text = "org.jacoco.agent"
                elemtree.SubElement(dependency_elem, "version").text = constants.JACOCO_VERSION
                elemtree.SubElement(dependency_elem, "classifier").text = "runtime"
                elemtree.SubElement(dependency_elem, "scope").text = "test"

            # add maven surefire plugin element, with argline configuration
            #       <plugin>
            #         <groupId>org.apache.maven.plugins</groupId>
            #         <artifactId>maven-surefire-plugin</artifactId>
            #         <configuration>
            #           <argLine>
            #              -javaagent:${settings.localRepository}/org/jacoco/org.jacoco.agent/0.8.13/org.jacoco.agent-0.8.13-runtime.jar=output=none,jmx=true
            #           </argLine>
            #         </configuration>
            #       </plugin>
            surefire_plugin = elemtree.SubElement(plugins, "plugin")
            elemtree.SubElement(surefire_plugin, "groupId").text = "org.apache.maven.plugins"
            elemtree.SubElement(surefire_plugin, "artifactId").text = "maven-surefire-plugin"
            surefire_config = elemtree.SubElement(surefire_plugin, "configuration")
            elemtree.SubElement(surefire_config, "argLine").text = \
                f"-javaagent:${{settings.localRepository}}/org/jacoco/org.jacoco.agent/{constants.JACOCO_VERSION}/org.jacoco.agent-{constants.JACOCO_VERSION}-runtime.jar=output=none,jmx=true"

        # write updated maven build file
        elemtree.indent(self.build_tree)
        self.build_tree.write(output_build_file, encoding="unicode")
        RichLog.info(f"Augmented build file written to {output_build_file}")

    def add_mutation_analysis_dependencies(self, output_build_file: str,
                                           target_tests: List[str] | None = None,
                                           excluded_tests: List[str] | None = None,
                                           target_classes: List[str] | None = None,
                                           mutation_operators: MutationOperators = MutationOperators.defaults):
        """
        Augments the Maven build file with Pitest dependencies and configuration for mutation analysis.

        Args:
            output_build_file (str): Path of output build file
            target_tests (list[str]): List of qualified test class names (glob-like patterns) to be included
                for computing mutation scores
            excluded_tests (list[str]): List of qualified test class names (glob-like patterns) to be excluded
                for computing mutation scores
            target_classes (list[str]): List of qualified app class names (glob-like patterns) to be included
                for mutant generation
            mutation_operators (MutationOperators): Mutation operators to be applied for mutant generation

        Returns:
            None
        """
        RichLog.info(f"Updating {self.project_root} build file for mutation analysis: {self.build_file}")
        root = self.build_tree.getroot()

        # locate or create <build>/<plugins>
        build = root.find("build", self.build_namespaces)
        if build is None:
            build = elemtree.SubElement(root, "build")
        plugins = build.find("plugins", self.build_namespaces)
        if plugins is None:
            plugins = elemtree.SubElement(build, "plugins")

        # check if PIT plugin already exists
        if self.__element_exists(parent_element=plugins, target_element="plugin",
                                 target_group_id="org.pitest", target_artifact_id="pitest-maven"):
            RichLog.info("PIT plugin already exists.")
            return

        # add PIT plugin
        pit_plugin = elemtree.SubElement(plugins, "plugin")
        group_id = elemtree.SubElement(pit_plugin, "groupId")
        group_id.text = "org.pitest"
        artifact_id = elemtree.SubElement(pit_plugin, "artifactId")
        artifact_id.text = "pitest-maven"
        version = elemtree.SubElement(pit_plugin, "version")
        version.text = MavenBuild.PITEST_MAVEN_VERSION

        # configure plugin with output formats and junit5 plugin
        config = elemtree.SubElement(pit_plugin, "configuration")
        output_formats = elemtree.SubElement(config, "outputFormats")
        format_ = elemtree.SubElement(output_formats, "outputFormat")
        format_.text = "HTML"
        format_ = elemtree.SubElement(output_formats, "outputFormat")
        format_.text = "XML"
        format_ = elemtree.SubElement(output_formats, "outputFormat")
        format_.text = "CSV"
        junit_plugin = elemtree.SubElement(config, "pluginConfiguration")
        junit_plugin_dep = elemtree.SubElement(junit_plugin, "plugin")
        junit_plugin_dep.text = "junit5"

        # set and failure and report configuration options
        elemtree.SubElement(config, "failWhenNoMutations").text = "false"
        elemtree.SubElement(config, "timestampedReports").text = "false"

        # set target tests and excluded tests
        if target_tests:
            target_tests_elem = elemtree.SubElement(config, "targetTests")
            for test_pattern in target_tests:
                elemtree.SubElement(target_tests_elem, "param").text = test_pattern
        # set target tests and excluded tests
        if excluded_tests:
            excluded_tests_elem = elemtree.SubElement(config, "excludedTestClasses")
            for test_pattern in excluded_tests:
                elemtree.SubElement(excluded_tests_elem, "param").text = test_pattern

        # configure target classes
        if target_classes:
            target_classes_elem = elemtree.SubElement(config, "targetClasses")
            for class_pattern in target_classes:
                elemtree.SubElement(target_classes_elem, "param").text = class_pattern

        # configure mutation operators
        elemtree.SubElement(config, "mutators").text = mutation_operators

        # add plugin dependencies
        dependencies = elemtree.SubElement(pit_plugin, "dependencies")
        plugin_dep = elemtree.SubElement(dependencies, "dependency")
        plugin_group = elemtree.SubElement(plugin_dep, "groupId")
        plugin_group.text = "org.pitest"
        plugin_artifact = elemtree.SubElement(plugin_dep, "artifactId")
        plugin_artifact.text = "pitest-junit5-plugin"
        plugin_version = elemtree.SubElement(plugin_dep, "version")
        plugin_version.text = MavenBuild.PITEST_JUNIT5_PLUGIN_VERSION

        # write updated build file
        elemtree.indent(self.build_tree)
        self.build_tree.write(output_build_file, encoding="unicode")
        RichLog.info(f"Augmented build file written to {output_build_file}")

    def __element_exists(self, parent_element: elemtree.Element, target_element: str,
                        target_group_id: str, target_artifact_id: str) -> bool:
        for element in parent_element.findall(target_element, self.build_namespaces):
            group_id = element.find("groupId", self.build_namespaces)
            artifact_id = element.find("artifactId", self.build_namespaces)
            if group_id is not None and artifact_id is not None:
                if group_id.text == target_group_id and artifact_id.text == target_artifact_id:
                    return True
        return False


if __name__ == "__main__":
    root_path = Path(__file__).parent.parent.parent.parent
    dataset_path = root_path.joinpath("resources/datasets")
    dataset_name = "commons-cli"
    augmented_pom_file = dataset_path.joinpath(dataset_name, "pom_mut.xml")
    builder = MavenBuild(project_root=str(dataset_path.joinpath(dataset_name)), build_file_name="pom.xml")
    builder.add_mutation_analysis_dependencies(output_build_file=str(augmented_pom_file))
