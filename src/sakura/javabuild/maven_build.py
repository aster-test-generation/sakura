# SpellChecker:ignore elemtree
"""
Maven Build Class

Vendored from aster-test-generation/javabuild
(branch feat/maven-module-scoped-runs, commit 5ee0c60) with imports rewired
to sakura's own constants and logger.
"""

import os
import subprocess
import sys
import xml.etree.ElementTree as elemtree
from pathlib import Path
from typing import Dict, List, Tuple
from xml.etree.ElementTree import ElementTree

from sakura.javabuild.abstract_build import AbstractBuild
from sakura.utils import constants
from sakura.utils.constants import MutationOperators
from sakura.utils.pretty.color_logger import RichLog


class MavenBuild(AbstractBuild):
    """This class performs all Maven build tasks"""

    MAVEN_CMD = "mvn.cmd" if sys.platform == "win32" else "mvn"

    # default maven options for running test-compile and test commands
    DEFAULT_MAVEN_OPTIONS = [
        "-Drat.skip",
        "-Dfindbugs.skip",
        "-Dcheckstyle.skip",
        "-Dpmd.skip=true",
        "-Dspotbugs.skip",
        "-Denforcer.skip",
        "-Dlicense.skip=true",
    ]

    # required maven dependencies for running generated tests
    WCA_TESTGEN_MAVEN_DEPENDENCIES = {
        "org.junit.jupiter": [
            {"artifact_id": "junit-jupiter-api", "version": "5.12.0", "scope": "test"}
        ],
        "org.mockito": [
            {"artifact_id": "mockito-core", "version": "5.12.0", "scope": "test"},
            {
                "artifact_id": "mockito-junit-jupiter",
                "version": "5.12.0",
                "scope": "test",
            },
        ],
    }

    # required maven dependencies for running generated tests
    WCA_TESTGEN_MAVEN_DEPENDENCIES_WITH_SPRING = {
        "org.junit.jupiter": [
            {"artifact_id": "junit-jupiter-api", "version": "5.12.0", "scope": "test"}
        ],
        "org.mockito": [
            {"artifact_id": "mockito-core", "version": "5.12.0", "scope": "test"},
            {
                "artifact_id": "mockito-junit-jupiter",
                "version": "5.12.0",
                "scope": "test",
            },
        ],
        "org.springframework": [
            {"artifact_id": "spring-test", "version": "5.3.36", "scope": "test"},
            {"artifact_id": "spring-web", "version": "6.1.6", "scope": "test"},
        ],
    }

    PITEST_MAVEN_VERSION = "1.19.4"
    PITEST_JUNIT5_PLUGIN_VERSION = "1.2.3"

    def __init__(
        self,
        project_root: str,
        build_file_name: str = "pom.xml",
        options: list = None,
        target_module: str = None,
        is_spring_project: bool = False,
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
        self.is_spring_project = is_spring_project
        self.build_system = "Maven"
        self.build_tree, self.build_namespaces = self.__parse_build_file()

    ######################################################################
    # PRIVATE METHODS
    ######################################################################

    def __parse_build_file(
        self, build_file: Path = None
    ) -> tuple[elemtree.ElementTree, dict]:
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
    def __qualified_child_tag(parent: elemtree.Element, tag: str) -> str:
        if parent.tag.startswith("{"):
            namespace = parent.tag[1:].split("}", maxsplit=1)[0]
            return f"{{{namespace}}}{tag}"
        return tag

    @classmethod
    def __sub_element(cls, parent: elemtree.Element, tag: str) -> elemtree.Element:
        return elemtree.SubElement(parent, cls.__qualified_child_tag(parent, tag))

    def __run_maven(
        self, command: List[str], timeout: int | None = 600
    ) -> subprocess.CompletedProcess[str]:
        """
        Executes a Maven command with optional timeout.

        Args:
            command: The Maven command to execute as a list of strings.
            timeout: Maximum time in seconds to wait for the command to complete.
                If None, waits indefinitely. Defaults to 600 seconds (10 minutes).

        Returns:
            A CompletedProcess object with the command results. If a timeout occurs,
            returns a CompletedProcess with returncode=-1 and a timeout error message.
        """
        RichLog.debug(f"Running command: {command}")
        try:
            return subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            RichLog.debug(f"Command timed out after {timeout} seconds: {command}")
            return subprocess.CompletedProcess(
                args=command,
                returncode=-1,
                stdout=f"[ERROR] Maven command timed out after {timeout} seconds\n",
                stderr="",
            )

    def __append_project_selection(self, command: List[str], also_make: bool) -> None:
        """
        Append Maven reactor selection flags for a targeted module in a multi-module build.

        Args:
            command: The Maven command argument list to mutate in-place.
            also_make: Whether to include `--also-make` to build dependent reactor modules.

        Returns:
            None. `command` is modified in-place.
        """
        if not self.target_module:
            return
        command.extend(["--projects", self.target_module])
        if also_make:
            command.append("--also-make")

    ################################################################################
    # IS MULTI MODULE PROJECT
    ################################################################################
    def is_multi_module_project(self) -> bool:
        """
        Checks whether this project is a multi-module project.

        Returns:
            bool: True if project is multi-module, False otherwise
        """
        root = self.build_tree.getroot()
        return bool(root.find("./modules", self.build_namespaces))

    def get_modules(self) -> Dict[str, Dict]:
        """
        Returns the hierarchy of modules for a multi-module project or empty dict for a project with
        no modules.

        Returns:
            Dict: Dictionary containing module hierarchy
        """
        module_hierarchy = dict()
        self.__get_module_hierarchy(
            pom_path=self.build_file, module_hierarchy=module_hierarchy
        )
        return module_hierarchy

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
                    pom_path=module_pom_path, module_hierarchy=module_hierarchy[module]
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
            for module in modules_element
            if module.text and module.text.strip()
        ]

    ################################################################################
    # PREPARE MODULES WITHOUT DEPENDENCY MODULE TEST COMPILATION OR EXECUTION
    ################################################################################
    def install_selected_projects_skip_tests_proc(
        self, timeout: int | None = 600
    ) -> subprocess.CompletedProcess[str]:
        """
        Prepare reactor dependencies so module-only tests can be executed without
        dependency module test compilation or execution (without --also-make)

        Args:
            timeout: Maximum time in seconds for the Maven command. Defaults to 600.
        """
        command = [
            self.MAVEN_CMD,
            "-f",
            str(self.build_file),
            *(self.DEFAULT_MAVEN_OPTIONS),
            "-Dmaven.test.skip=true",
            "-DskipTests=true",
            "-DskipITs=true",
            "install",
            "-Dstyle.color=never",
            (
                "-Dspring-javaformat.skip=true"
                if self.is_spring_project
                else "-Dspring-javaformat.skip=false"
            ),
        ]

        self.__append_project_selection(command, also_make=True)
        return self.__run_maven(command, timeout=timeout)

    def install_selected_projects_skip_tests(self) -> str:
        return self.install_selected_projects_skip_tests_proc().stdout

    ################################################################################
    # GET PARENT MODULE PATH
    ################################################################################
    def get_parent_module_path(self) -> Path | None:
        """
        If this project has a parent module, returns the path of the parent; returns None otherwise.

        Returns:
            Path: Path object for the parent path; None if there is no parent module
        """
        root = self.build_tree.getroot()
        if root is None:
            return None
        parent_element = root.find("./parent", self.build_namespaces)
        if parent_element is None:
            return None

        relative_path_element = parent_element.find(
            "./relativePath", self.build_namespaces
        )
        if relative_path_element is None:
            relative_path = Path("../pom.xml")
        else:
            relative_path_text = (relative_path_element.text or "").strip()
            if not relative_path_text:
                return None
            relative_path = Path(relative_path_text)

        parent_build_file = self.build_file.parent.joinpath(relative_path).resolve()
        if parent_build_file.is_dir():
            # Maven resolves a relativePath pointing at a directory to its pom.xml
            parent_build_file = parent_build_file.joinpath("pom.xml")
        if not parent_build_file.is_file():
            return None

        parent_tree, parent_namespaces = self.__parse_build_file(parent_build_file)
        parent_dir = parent_build_file.parent
        child_project_path = self.project_root.resolve()
        for parent_module_element in parent_tree.findall(
            "./modules/module", parent_namespaces
        ):
            module_text = (parent_module_element.text or "").strip()
            if not module_text:
                continue
            module_path = parent_dir.joinpath(module_text).resolve()
            if module_path == child_project_path:
                return parent_dir
        return None

    def __get_version_from_tree_element(self, root, match_str):
        """
        Returns the version number based on match_str criteria

        """
        found = False
        version = ""
        for entry in match_str:
            version_element = root.find(entry, self.build_namespaces)
            if version_element is not None:
                version = version_element.text.split(".")[-1]
                found = True
                break
        return version, found

    ################################################################################
    # GET JAVA VERSION
    ################################################################################
    def get_java_version(self) -> str:
        """
        Checks for the java compiler version in the pom.xml and returns it

        Returns:
            int: java compiler version
        """
        version = constants.MAVEN_DEFAULT_VERSION
        root = self.build_tree.getroot()
        found = False
        # Example Type1
        # <maven.compiler.target>1.10</maven.compiler.target> or
        # <maven.compiler.release>8</maven.compiler.release> or
        # <maven.compiler.source>8</maven.compiler.source> or
        # <java.version>11</java.version>
        match_str = [
            "./properties/maven.compiler.target",
            "./properties/maven.compiler.release",
            "./properties/maven.compiler.source",
            "./properties/java.version",
            "./properties/jdk.version",
        ]
        version_type1, found = self.__get_version_from_tree_element(root, match_str)
        if found:
            version = version_type1

        # Example Type2
        # <build>
        #   <plugins>
        #     <plugin>
        #       <artifactId>maven-compiler-plugin</artifactId>
        #       <configuration>
        #         <target>1.8</target> (or) <release>8</release> (or) <source>8</source>
        #       </configuration>
        #     </plugin>
        #   </plugins>
        # </build>
        if not found:
            for plugin_element in root.findall(
                "./build/plugins/plugin", self.build_namespaces
            ):
                artifact_id_element = plugin_element.find(
                    "./artifactId", self.build_namespaces
                )
                if artifact_id_element is not None:
                    if artifact_id_element.text == "maven-compiler-plugin":
                        match1_str = [
                            "./configuration/target",
                            "./configuration/release",
                            "./configuration/source",
                        ]
                        version_type2, found = self.__get_version_from_tree_element(
                            plugin_element, match1_str
                        )
                        if found:
                            version = version_type2
                            break
        return version

    ################################################################################
    # ADD WCA TEST DEPENDENCIES
    ################################################################################
    def add_wca_test_dependencies(
        self, output_build_file: str = None, is_add_spring_dependency: bool = False
    ) -> None:
        """
        Checks the maven build file for dependencies required for running the generated tests, adds the missing
        dependencies, and writes out the augmented build file.

        Args:
            is_add_spring_dependency: if the project is spring and set by the user
            output_build_file: Name of the output build file (by default, the original build file is overwritten)

        Returns:

        """
        # if output build file not specified, overwrite the original build file
        if output_build_file is None:
            output_build_file = self.build_file_name

        # parse either the root or the module build file
        if self.target_module:
            tree, namespaces = self.__parse_build_file(
                build_file=self.project_root.joinpath(self.target_module).joinpath(
                    output_build_file
                )
            )
        else:
            namespaces = self.build_namespaces
            tree = self.build_tree
        root = tree.getroot()
        assert root is not None

        # get "dependencies" element from build file; add it if it does not exist
        deps_element = root.find("./dependencies", namespaces)
        if deps_element is None:
            deps_element = self.__sub_element(root, "dependencies")

        dependency = (
            self.WCA_TESTGEN_MAVEN_DEPENDENCIES_WITH_SPRING
            if is_add_spring_dependency
            else self.WCA_TESTGEN_MAVEN_DEPENDENCIES
        )

        # check for each required dependency and add it to the build file if needed
        for group_id, required_group_deps in dependency.items():
            for required_dep in required_group_deps:
                artifact_id = required_dep["artifact_id"]
                if self.__element_exists(
                    parent_element=deps_element,
                    target_element="dependency",
                    target_group_id=group_id,
                    target_artifact_id=artifact_id,
                ):
                    continue
                elif artifact_id == "junit-jupiter-api":
                    if self.__element_exists(
                        parent_element=deps_element,
                        target_element="dependency",
                        target_group_id=group_id,
                        target_artifact_id="junit-jupiter",
                    ):
                        continue

                dependency = self.__sub_element(deps_element, "dependency")
                self.__sub_element(dependency, "groupId").text = group_id
                self.__sub_element(dependency, "artifactId").text = required_dep[
                    "artifact_id"
                ]
                self.__sub_element(dependency, "version").text = required_dep["version"]
                self.__sub_element(dependency, "scope").text = required_dep["scope"]

        # write out augmented build file
        if self.target_module:
            augmented_build_file = self.project_root.joinpath(
                self.target_module
            ).joinpath(output_build_file)
        else:
            augmented_build_file = self.project_root.joinpath(output_build_file)
        elemtree.indent(tree)
        tree.write(augmented_build_file, encoding="unicode")

    ################################################################################
    # ADD CODE COVERAGE DEPENDENCIES
    ################################################################################
    def add_code_coverage_dependencies(
        self, output_build_file: str, add_java_agent: bool = False
    ):
        """
        Augments a given Maven build file with Jacoco dependencies and configuration for
        collecting code coverage.

        Args:
            output_build_file (str): Path of output build file
            add_java_agent (bool): Add java agent as arg-line option for maven surefire plugin

        Returns: Path to the augmented build file

        """
        RichLog.debug(f"Updating {self.project_root} build file: {self.build_file}")
        root = self.build_tree.getroot()
        assert root is not None

        # locate or create <build>/<plugins>
        build = root.find("build", self.build_namespaces)
        if build is None:
            build = self.__sub_element(root, "build")
        plugins = build.find("plugins", self.build_namespaces)
        if plugins is None:
            plugins = self.__sub_element(build, "plugins")

        # add and configure jacoco plugin if it does not exist
        if not self.__element_exists(
            parent_element=plugins,
            target_element="plugin",
            target_group_id="org.jacoco",
            target_artifact_id="jacoco-maven-plugin",
        ):
            jacoco_plugin = self.__sub_element(plugins, "plugin")
            self.__sub_element(jacoco_plugin, "groupId").text = "org.jacoco"
            self.__sub_element(jacoco_plugin, "artifactId").text = "jacoco-maven-plugin"
            self.__sub_element(jacoco_plugin, "version").text = constants.JACOCO_VERSION

            # configure plugin
            config = self.__sub_element(jacoco_plugin, "configuration")
            self.__sub_element(
                config, "destFile"
            ).text = constants.MAVEN_JACOCO_COV_FILE
            self.__sub_element(
                config, "dataFile"
            ).text = constants.MAVEN_JACOCO_COV_FILE
            self.__sub_element(
                config, "outputDirectory"
            ).text = constants.MAVEN_COV_REPORT_DIR

            # add executions element
            executions = self.__sub_element(jacoco_plugin, "executions")

            # execution element for "prepare-agent" goal
            execution = self.__sub_element(executions, "execution")
            goals = self.__sub_element(execution, "goals")
            self.__sub_element(goals, "goal").text = "prepare-agent"

            # execution element for "report" goal
            execution = self.__sub_element(executions, "execution")
            self.__sub_element(execution, "id").text = "report"
            self.__sub_element(execution, "phase").text = "test"
            goals = self.__sub_element(execution, "goals")
            self.__sub_element(goals, "goal").text = "report"

        if add_java_agent:
            # locate or create <dependencies>
            dependencies = root.find("dependencies", self.build_namespaces)
            if dependencies is None:
                dependencies = self.__sub_element(root, "dependencies")

            # if jacoco agent dependency does not exist, add it
            if not self.__element_exists(
                parent_element=dependencies,
                target_element="dependency",
                target_group_id="org.jacoco",
                target_artifact_id="org.jacoco.agent",
            ):
                dependency_elem = self.__sub_element(dependencies, "dependency")
                self.__sub_element(dependency_elem, "groupId").text = "org.jacoco"
                self.__sub_element(
                    dependency_elem, "artifactId"
                ).text = "org.jacoco.agent"
                self.__sub_element(
                    dependency_elem, "version"
                ).text = constants.JACOCO_VERSION
                self.__sub_element(dependency_elem, "classifier").text = "runtime"
                self.__sub_element(dependency_elem, "scope").text = "test"

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
            surefire_plugin = self.__find_element(
                parent_element=plugins,
                target_element="plugin",
                target_group_id="org.apache.maven.plugins",
                target_artifact_id="maven-surefire-plugin",
                allow_missing_group_id=True,
            )
            if surefire_plugin is None:
                surefire_plugin = self.__sub_element(plugins, "plugin")
                self.__sub_element(
                    surefire_plugin, "groupId"
                ).text = "org.apache.maven.plugins"
                self.__sub_element(
                    surefire_plugin, "artifactId"
                ).text = "maven-surefire-plugin"

            surefire_config = surefire_plugin.find(
                "configuration", self.build_namespaces
            )
            if surefire_config is None:
                surefire_config = self.__sub_element(surefire_plugin, "configuration")
            arg_line = surefire_config.find("argLine", self.build_namespaces)
            if arg_line is None:
                arg_line = self.__sub_element(surefire_config, "argLine")
            java_agent_arg = (
                f"-javaagent:${{settings.localRepository}}/org/jacoco/"
                f"org.jacoco.agent/{constants.JACOCO_VERSION}/"
                f"org.jacoco.agent-{constants.JACOCO_VERSION}-runtime.jar="
                "output=none,jmx=true"
            )
            existing_arg_line = (arg_line.text or "").strip()
            if java_agent_arg not in existing_arg_line:
                arg_line.text = f"{existing_arg_line} {java_agent_arg}".strip()

        # write updated maven build file
        elemtree.indent(self.build_tree)
        self.build_tree.write(output_build_file, encoding="unicode")
        RichLog.debug(f"Augmented build file written to {output_build_file}")

    ################################################################################
    # ADD MUTATION ANALYSIS DEPENDENCIES
    ################################################################################
    def add_mutation_analysis_dependencies(
        self,
        output_build_file: str,
        target_tests: List[str] | None = None,
        excluded_tests: List[str] | None = None,
        target_classes: List[str] | None = None,
        mutation_operators: MutationOperators = MutationOperators.defaults,
    ):
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
        RichLog.debug(
            f"Updating {self.project_root} build file for mutation analysis: {self.build_file}"
        )
        root = self.build_tree.getroot()
        assert root is not None

        # locate or create <build>/<plugins>
        build = root.find("build", self.build_namespaces)
        if build is None:
            build = self.__sub_element(root, "build")
        plugins = build.find("plugins", self.build_namespaces)
        if plugins is None:
            plugins = self.__sub_element(build, "plugins")

        # check if PIT plugin already exists
        if self.__element_exists(
            parent_element=plugins,
            target_element="plugin",
            target_group_id="org.pitest",
            target_artifact_id="pitest-maven",
        ):
            RichLog.info("PIT plugin already exists.")
            elemtree.indent(self.build_tree)
            self.build_tree.write(output_build_file, encoding="unicode")
            return

        # add PIT plugin
        pit_plugin = self.__sub_element(plugins, "plugin")
        group_id = self.__sub_element(pit_plugin, "groupId")
        group_id.text = "org.pitest"
        artifact_id = self.__sub_element(pit_plugin, "artifactId")
        artifact_id.text = "pitest-maven"
        version = self.__sub_element(pit_plugin, "version")
        version.text = MavenBuild.PITEST_MAVEN_VERSION

        # configure plugin with output formats and junit5 plugin
        config = self.__sub_element(pit_plugin, "configuration")
        output_formats = self.__sub_element(config, "outputFormats")
        format_ = self.__sub_element(output_formats, "outputFormat")
        format_.text = "HTML"
        format_ = self.__sub_element(output_formats, "outputFormat")
        format_.text = "XML"
        format_ = self.__sub_element(output_formats, "outputFormat")
        format_.text = "CSV"
        junit_plugin = self.__sub_element(config, "pluginConfiguration")
        junit_plugin_dep = self.__sub_element(junit_plugin, "plugin")
        junit_plugin_dep.text = "junit5"

        # set and failure and report configuration options
        self.__sub_element(config, "failWhenNoMutations").text = "false"
        self.__sub_element(config, "timestampedReports").text = "false"

        # set target tests and excluded tests
        if target_tests:
            target_tests_elem = self.__sub_element(config, "targetTests")
            for test_pattern in target_tests:
                self.__sub_element(target_tests_elem, "param").text = test_pattern
        # set target tests and excluded tests
        if excluded_tests:
            excluded_tests_elem = self.__sub_element(config, "excludedTestClasses")
            for test_pattern in excluded_tests:
                self.__sub_element(excluded_tests_elem, "param").text = test_pattern

        # configure target classes
        if target_classes:
            target_classes_elem = self.__sub_element(config, "targetClasses")
            for class_pattern in target_classes:
                self.__sub_element(target_classes_elem, "param").text = class_pattern

        # configure mutation operators
        self.__sub_element(config, "mutators").text = mutation_operators

        # add plugin dependencies
        dependencies = self.__sub_element(pit_plugin, "dependencies")
        plugin_dep = self.__sub_element(dependencies, "dependency")
        plugin_group = self.__sub_element(plugin_dep, "groupId")
        plugin_group.text = "org.pitest"
        plugin_artifact = self.__sub_element(plugin_dep, "artifactId")
        plugin_artifact.text = "pitest-junit5-plugin"
        plugin_version = self.__sub_element(plugin_dep, "version")
        plugin_version.text = MavenBuild.PITEST_JUNIT5_PLUGIN_VERSION

        # write updated build file
        elemtree.indent(self.build_tree)
        self.build_tree.write(output_build_file, encoding="unicode")
        RichLog.debug(f"Augmented build file written to {output_build_file}")

    def __find_element(
        self,
        parent_element: elemtree.Element,
        target_element: str,
        target_group_id: str,
        target_artifact_id: str,
        allow_missing_group_id: bool = False,
    ) -> elemtree.Element | None:
        element_tag = self.__qualified_child_tag(parent_element, target_element)
        for element in parent_element.findall(element_tag):
            group_id = element.find(self.__qualified_child_tag(element, "groupId"))
            artifact_id = element.find(
                self.__qualified_child_tag(element, "artifactId")
            )
            if artifact_id is None or artifact_id.text != target_artifact_id:
                continue
            if group_id is None and allow_missing_group_id:
                return element
            if group_id is not None and group_id.text == target_group_id:
                return element
        return None

    def __element_exists(
        self,
        parent_element: elemtree.Element,
        target_element: str,
        target_group_id: str,
        target_artifact_id: str,
    ) -> bool:
        return (
            self.__find_element(
                parent_element=parent_element,
                target_element=target_element,
                target_group_id=target_group_id,
                target_artifact_id=target_artifact_id,
            )
            is not None
        )

    ################################################################################
    # COMPILE APPLICATION
    ################################################################################
    def is_compile_application(self) -> bool:
        """This method calls maven to compile the application

        Returns:
            str: The output of the compile delimited by new lines
        """
        command = [
            self.MAVEN_CMD,
            "-f",
            os.path.join(self.project_root, self.build_file_name),
            *(self.DEFAULT_MAVEN_OPTIONS),
            "clean",
            "test-compile",
            "-Dstyle.color=never",
            (
                "-Dspring-javaformat.skip=true"
                if self.is_spring_project
                else "-Dspring-javaformat.skip=false"
            ),
        ]
        # if target module is specified (for a multi-module project), compile the specified module
        # and its dependencies
        if self.target_module:
            command.extend(["--projects", self.target_module, "--also-make"])

        # run command and return the output
        RichLog.debug(f"Running command: {command}")
        try:
            response = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            if "build success" not in str(response.stdout).lower():
                return False
            return True
        except subprocess.CalledProcessError:
            return False

    ################################################################################
    # COMPILE TESTS
    ################################################################################
    def compile_tests_proc(
        self,
        pre_compile_build: bool = False,
        also_make: bool = True,
        timeout: int | None = 600,
    ) -> subprocess.CompletedProcess[str]:
        """
        Compile test sources with Maven and return the full subprocess result.

        Args:
            pre_compile_build: If True, run `clean test-compile`; otherwise run
                `clean compile compiler:testCompile`.
            also_make: When targeting a module, whether to build required reactor
                modules via `--also-make`.
            timeout: Maximum time in seconds for the Maven command. Defaults to 600.

        Returns:
            The CompletedProcess for the Maven invocation (stdout/stderr/returncode).
        """
        if pre_compile_build:
            command = [
                self.MAVEN_CMD,
                "-f",
                os.path.join(self.project_root, self.build_file_name),
                *(self.DEFAULT_MAVEN_OPTIONS),
                "clean",
                "test-compile",
                "-Dstyle.color=never",
                (
                    "-Dspring-javaformat.skip=true"
                    if self.is_spring_project
                    else "-Dspring-javaformat.skip=false"
                ),
                "-Dmaven.compiler.compilerArgs=-Xmaxerrs 20000",
            ]
        else:
            command = [
                self.MAVEN_CMD,
                "-f",
                os.path.join(self.project_root, self.build_file_name),
                *(self.DEFAULT_MAVEN_OPTIONS),
                "clean",
                "compile",
                "compiler:testCompile",
                "-Dstyle.color=never",
                (
                    "-Dspring-javaformat.skip=true"
                    if self.is_spring_project
                    else "-Dspring-javaformat.skip=false"
                ),
                "-Dmaven.compiler.compilerArgs=-Xmaxerrs 20000",
            ]

        self.__append_project_selection(command, also_make=also_make)
        return self.__run_maven(command, timeout=timeout)

    def compile_tests(
        self,
        pre_compile_build: bool = False,
        also_make: bool = True,
        timeout: int | None = 600,
    ) -> str:
        """
        Compile test sources with Maven and return stdout as a string.

        Args:
            pre_compile_build: If True, run `clean test-compile`; otherwise run
                `clean compile compiler:testCompile`.
            also_make: When targeting a module, whether to build required reactor
                modules via `--also-make`.
            timeout: Maximum time in seconds for the Maven command. Defaults to 600.

        Returns:
            Maven stdout for the compilation command.
        """
        return self.compile_tests_proc(
            pre_compile_build=pre_compile_build, also_make=also_make, timeout=timeout
        ).stdout

    ################################################################################
    # RUN TESTS
    ################################################################################
    def run_tests_proc(
        self,
        target_tests: str | None = None,
        build_file: str | None = None,
        also_make: bool = True,
        timeout: int | None = 600,
    ) -> subprocess.CompletedProcess[str]:
        """
        Run Maven tests and return the full subprocess result.

        Args:
            target_tests: Surefire test selector (e.g., "MyTest", "MyTest#method",
                or glob patterns depending on the provider). If None, run all tests.
            build_file: Path to a Maven POM to use instead of the default.
            also_make: When targeting a module, whether to build required reactor
                modules via `--also-make`.
            timeout: Maximum time in seconds for the Maven command. Defaults to 600.

        Returns:
            The CompletedProcess for the Maven invocation (stdout/stderr/returncode).
        """
        command = [
            self.MAVEN_CMD,
            "-f",
            build_file if build_file else str(self.build_file),
            *(self.DEFAULT_MAVEN_OPTIONS),
            *(["-Dtest=" + target_tests] if target_tests else []),
            "clean",
            "test",
            "-Dstyle.color=never",
            "-DskipTests=false",
            "-Dmaven.test.skip=false",
            "-Dmaven.test.failure.ignore=true",
            (
                "-Dspring-javaformat.skip=true"
                if self.is_spring_project
                else "-Dspring-javaformat.skip=false"
            ),
        ]

        self.__append_project_selection(command, also_make=also_make)
        return self.__run_maven(command, timeout=timeout)

    def run_tests(
        self,
        target_tests: str | None = None,
        build_file: str | None = None,
        also_make: bool = True,
        timeout: int | None = 600,
    ) -> str:
        """
        Run Maven tests and return stdout as a string.

        Args:
            target_tests: Surefire test selector (see `run_tests_proc`).
            build_file: Path to a Maven POM to use instead of the default.
            also_make: When targeting a module, whether to build required reactor
                modules via `--also-make`.
            timeout: Maximum time in seconds for the Maven command. Defaults to 600.

        Returns:
            Maven stdout for the test run (newline-delimited).
        """
        return self.run_tests_proc(
            target_tests=target_tests,
            build_file=build_file,
            also_make=also_make,
            timeout=timeout,
        ).stdout

    ################################################################################
    # RUN SANITATION RECIPES
    ################################################################################
    def run_sanitization_recipes(self):
        """Runs the recipes for sanitization"""
        raise NotImplementedError

    ################################################################################
    # FIND COMPILE ERRORS
    ################################################################################
    def find_compile_errors(
        self, class_name: str, compiler_output: str
    ) -> Tuple[List[int], Dict[int, str]]:
        """Finds compiler errors based on Maven output

        Args:
            class_name (str): The nae of the class being compiled
            compiler_output (str): The output of the compile

        Returns:
            list: A list of line numbers that have errors
            dict: A dictionary with the line number as the key and the line as the value
        """
        RichLog.debug(f"--> find_compile_errors({class_name})...")

        problem_line = []
        error_line = {}
        for line in compiler_output.splitlines():
            # pylint: disable=line-too-long
            # noqa: E501
            # Example error lines to parse:
            # [ERROR] /aster/resources/datasets/modresorts/src/test/java/com/acme/modres/WCA_DefaultWeatherData_Test.java:[54,42] ';' expected
            # [ERROR] /aster/resources/datasets/modresorts/src/test/java/com/acme/modres/WCA_DefaultWeatherData_Test.java:[54,52] not a statement
            if "[ERROR]" in line and class_name + ".java:[" in line:
                RichLog.debug(f">>> ERROR in line: {line}")
                line_number = (
                    int(line.split(class_name + ".java:[")[1].split(",")[0]) - 1
                )
                problem_line.append(line_number)
                error_line[line_number] = line

        problem_line = list(set(problem_line))

        RichLog.debug(f">>> Found {str(len(problem_line))} error(s).")
        if error_line:
            RichLog.debug(f"<<< Returning: problem_line={problem_line}")
            RichLog.debug(f"<<< Returning: error_line={error_line}")

        return problem_line, error_line

    ################################################################################
    # REMOVE FUNCTION WITH ERRORS
    ################################################################################
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
        RichLog.debug(f"--> remove_functions_with_errors({class_name})...")
        remove_functions = []
        for line in compiler_output.splitlines():
            if "[ERROR]" in line and class_name + "." in line and "compilation" in line:
                # Need examples of the line to parse
                RichLog.debug(f">>> ERROR in line: {line}")
                test_function_name = (line.split(class_name + ".")[1]).split(":")[0]
                if (
                    test_function_name != ""
                    and test_function_name != "java"
                    and test_function_name not in remove_functions
                ):
                    remove_functions.append(test_function_name)
                    error_and_code_body[test_function_name] = line
        RichLog.debug(
            f">>> Found {str(len(remove_functions))} functions with error(s)."
        )
        if remove_functions:
            RichLog.debug(f"<<< Returning: {remove_functions}")

        return remove_functions

    ################################################################################
    # FIND RUNTIME ERROR FOR METHOD
    ################################################################################
    def find_runtime_error_for_method(
        self, compiler_output: str, test_class_name: str, method_name: str
    ) -> Dict[int, str]:
        """Searches compiler output for errors and returns the line number and error message

        Args:
            compiler_output (str): Output of the compile step
            test_class_name (str): Name of the class under test
            method_name (str): Name of the method under test

        Returns:
            Dict[int, str]: The line number as the key, and error line as the value
        """
        RichLog.debug(
            f"--> find_runtime_error_for_method({test_class_name}.{method_name})..."
        )
        problem_line = {}
        capture = False
        line_number = -1
        for line in compiler_output.splitlines():
            if "[ERROR]" in line and test_class_name + "." in line:
                # pylint: disable=line-too-long
                # noqa: E501
                # Example error lines with and without line numbers
                # [ERROR] com.acme.modres.WCA_DefaultWeatherData_Fixed_Test_Temp.test1_bmYG0 -- Time elapsed: 0.072 s <<< ERROR!
                # [ERROR]   WCA_DefaultWeatherData_Fixed_Test_Temp.test1_bmYG0:235 » UnsupportedOperation City is invalid. It must be one of [Ljava.lang.String;@26a4940c
                # [ERROR] /aster/resources/datasets/modresorts/src/test/java/com/acme/modres/WCA_DefaultWeatherData_Fixed_Test_Temp.java:[265,35] cannot find symbol
                RichLog.debug(f">>> ERROR in line: {line}")
                if test_class_name + "." + method_name + ":" in line:
                    potential_line_number = (
                        line.split(test_class_name + "." + method_name + ":")[1]
                        .split(" ")[0]
                        .strip()
                    )
                    if potential_line_number is not None:
                        line_number = int(potential_line_number)
                    if line_number != -1:
                        problem_line[line_number] = line
                        capture = True
                        continue

            # This signifies the end of the maven output
            if "Tests run:" in line and capture:
                RichLog.debug(f">>> Tests run in line: {line}")
                capture = False
                return problem_line

            if capture and "[ERROR]" in line:
                RichLog.debug(f">>> Capture line: {line}")
                if line_number != -1:
                    problem_line[line_number] = problem_line[line_number] + line

        RichLog.debug(
            f"<-- find_runtime_error_for_method({test_class_name}.{method_name})..."
        )
        if capture:
            return problem_line
        return {}

    ################################################################################
    # FIND RUNTIME ERROR FOR CLASS
    ################################################################################
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
        RichLog.debug(f"--> find_runtime_error_for_class({test_class_name})...")
        problem_line = {}
        capture = False
        line_number = -1
        method_name = ""
        temp = {}
        for line in compiler_output.splitlines():
            if "[ERROR]" in line and test_class_name + "." in line:
                # pylint: disable=line-too-long
                # noqa: E501
                # Example error lines with and without line numbers
                # [ERROR] com.acme.modres.WCA_DefaultWeatherData_Fixed_Test.test2_UkpZ1 -- Time elapsed: 0.053 s <<< ERROR!
                # [ERROR]   WCA_DefaultWeatherData_Fixed_Test.getDefaultWeatherDataForBarcelonaTest_WPRe5:383 » NullPointer Cannot invoke "java.io.InputStream.read(byte[])" because "inputStream" is null
                # [ERROR]   WCA_DefaultWeatherData_Fixed_Test.getDefaultWeatherDataTest1_IXyG0:390 » UnsupportedOperation City is invalid. It must be one of [Ljava.lang.String;@3fa6ac62
                RichLog.debug(f">>> ERROR in line: {line}")
                if capture and method_name != "":
                    temp[method_name] = problem_line
                    problem_line = {}
                    capture = False
                method_name = line.split(test_class_name + ".")[1].split(":")[0].strip()
                if " " in method_name.strip():
                    method_name = method_name.split(" ")[0]
                if test_class_name + "." + method_name + ":" in line:
                    line_number = (
                        line.split(test_class_name + "." + method_name + ":")[1]
                        .split(" ")[0]
                        .strip()
                    )
                    if line_number != "":
                        if "," in line_number:
                            line_number = int(line_number.split(",")[0].lstrip("["))
                        else:
                            try:
                                line_number = int(line_number)
                            except ValueError:
                                line_number = -1
                    # line = ' '.join(line)
                    if line_number != -1:
                        problem_line[line_number] = line
                        capture = True

            # This signifies the end of the maven output
            if "Tests run:" in line and capture:
                RichLog.debug(f">>> Tests run in line: {line}")
                capture = False
                if method_name != "":
                    temp[method_name] = problem_line
                    problem_line = {}
                    method_name = ""

            if capture and ("ERROR" not in line and "INFO" not in line):
                RichLog.debug(f">>> Capture line: {line}")
                if line_number != -1:
                    if line not in problem_line[line_number]:
                        problem_line[line_number] = problem_line[line_number] + line
        if capture and method_name != "":
            temp[method_name] = problem_line
        if len(temp) > 0:
            return temp
        return {}

    ################################################################################
    # GET CLASS NAMES WITH COMPILER ERRORS
    ################################################################################
    def get_class_names_with_compile_errors(self, compiler_output: str) -> List[str]:
        """Returns a list of class names that had compiler errors

        Args:
            compiler_output (str): Output from a test compile

        Returns:
            List[str]: A list of class names that had errors when compiled
        """
        error_classes = []
        for line in compiler_output.splitlines():
            if "[ERROR]" in line and ".java:[" in line:
                error_classes.append(line.split(":[")[0].split("[ERROR] ")[-1])

        return list(set(error_classes))
