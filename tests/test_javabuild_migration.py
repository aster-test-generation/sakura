"""
Migration stress tests for the vendored javabuild and re-aster code.

These tests pin the full contract sakura relies on from the vendored
sakura.javabuild package and the vendored JaCoCo test-watcher template
(sakura.utils.coverage.jacoco_test_watcher), so any drift from the
behavior of the original external dependencies is caught here:

- BuildFactory dispatch semantics
- The exact MavenBuild API surface and defaults consumed across sakura
- Maven command construction (module scoping, --also-make, -Dtest,
  custom build files, timeout handling)
- POM introspection (multi-module hierarchies, namespaced and
  non-namespaced POMs, java version detection variants)
- Coverage POM mutation (JaCoCo plugin/agent injection and idempotency)
- The TEST_CODE template contract shared with IndividualTestCoverage
- A real end-to-end per-test coverage run against spring-petclinic that
  drives the entire migrated reaster+javabuild pipeline
"""

import copy
import inspect
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from sakura.javabuild import AbstractBuild, BuildFactory, MavenBuild
from sakura.utils import constants
from sakura.utils.compilation.maven import JavaMavenCompilation
from sakura.utils.coverage.individual_test_coverage import (
    JACOCO_CLI_JAR,
    JACOCO_TEST_FOLDER,
    TEST_WATCHER_CLASS_NAME,
    IndividualTestCoverage,
)
from sakura.utils.coverage.jacoco_test_watcher import TEST_CODE
from sakura.utils.execution.maven import JavaMavenExecution


NAMESPACED_POM = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>demo</artifactId>
  <version>1.0.0</version>
  <properties>
    <maven.compiler.target>17</maven.compiler.target>
  </properties>
</project>
"""

PLAIN_POM = """<?xml version="1.0" encoding="UTF-8"?>
<project>
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>demo</artifactId>
  <version>1.0.0</version>
</project>
"""


def _compiler_pom(properties: str = "", configuration: str = "") -> str:
    properties_xml = f"<properties>{properties}</properties>" if properties else ""
    plugin_xml = (
        f"""<build>
    <plugins>
      <plugin>
        <artifactId>maven-compiler-plugin</artifactId>
        <configuration>{configuration}</configuration>
      </plugin>
    </plugins>
  </build>"""
        if configuration
        else ""
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>demo</artifactId>
  <version>1.0.0</version>
  {properties_xml}
  {plugin_xml}
</project>
"""


def _write_pom(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    pom = directory.joinpath("pom.xml")
    pom.write_text(body)
    return pom


def _pom_project(tmp_path: Path, body: str = NAMESPACED_POM) -> Path:
    project_root = tmp_path.joinpath("demo")
    _write_pom(project_root, body)
    return project_root


def _parent_pom(module: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>parent</artifactId>
  <version>1.0.0</version>
  <packaging>pom</packaging>
  <modules>
    <module>{module}</module>
  </modules>
</project>
"""


def _child_pom(display_name: str, relative_path: str | None = None) -> str:
    if relative_path is None:
        relative_path_element = ""
    elif relative_path:
        relative_path_element = f"    <relativePath>{relative_path}</relativePath>\n"
    else:
        relative_path_element = "    <relativePath/>\n"

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <parent>
    <groupId>com.example</groupId>
    <artifactId>parent</artifactId>
    <version>1.0.0</version>
{relative_path_element}  </parent>
  <artifactId>child</artifactId>
  <name>{display_name}</name>
</project>
"""


def _multimodule_project(tmp_path: Path) -> Path:
    """parent -> {web -> {web-api}, core}"""
    root = tmp_path.joinpath("parent")
    _write_pom(
        root,
        """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>parent</artifactId>
  <version>1.0.0</version>
  <packaging>pom</packaging>
  <modules>
    <module>web</module>
    <module>core</module>
  </modules>
</project>
""",
    )
    _write_pom(
        root.joinpath("web"),
        """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <parent>
    <groupId>com.example</groupId>
    <artifactId>parent</artifactId>
    <version>1.0.0</version>
  </parent>
  <artifactId>web</artifactId>
  <name>web</name>
  <packaging>pom</packaging>
  <modules>
    <module>web-api</module>
  </modules>
</project>
""",
    )
    _write_pom(
        root.joinpath("web", "web-api"),
        """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <artifactId>web-api</artifactId>
  <groupId>com.example</groupId>
  <version>1.0.0</version>
</project>
""",
    )
    _write_pom(
        root.joinpath("core"),
        """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <parent>
    <groupId>com.example</groupId>
    <artifactId>parent</artifactId>
    <version>1.0.0</version>
  </parent>
  <artifactId>core</artifactId>
  <name>core</name>
</project>
""",
    )
    return root


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _elements(root: ET.Element, name: str) -> list:
    return [e for e in root.iter() if _local_name(e.tag) == name]


def _child_text(element: ET.Element, name: str):
    for child in element:
        if _local_name(child.tag) == name:
            return child.text
    return None


def _find_plugin(tree_root: ET.Element, artifact_id: str) -> list:
    return [
        plugin
        for plugin in _elements(tree_root, "plugin")
        if _child_text(plugin, "artifactId") == artifact_id
    ]


@pytest.fixture
def captured_maven(monkeypatch):
    """Intercept every Maven invocation, recording the exact command."""
    calls = []

    def fake_run(command, **kwargs):
        calls.append({"command": list(command), "kwargs": kwargs})
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout="[INFO] BUILD SUCCESS\n", stderr=""
        )

    monkeypatch.setattr("sakura.javabuild.maven_build.subprocess.run", fake_run)
    return calls


class TestBuildFactory:
    def test_create_maven_returns_maven_build(self, tmp_path):
        project_root = _pom_project(tmp_path)
        builder = BuildFactory.create("maven", str(project_root))
        assert isinstance(builder, MavenBuild)
        assert isinstance(builder, AbstractBuild)

    def test_create_is_case_insensitive(self, tmp_path):
        project_root = _pom_project(tmp_path)
        builder = BuildFactory.create("MaVeN", str(project_root))
        assert isinstance(builder, MavenBuild)

    def test_create_gradle_not_vendored(self):
        with pytest.raises(NotImplementedError, match="GradleBuild was not vendored"):
            BuildFactory.create("gradle", "/nonexistent")

    def test_create_unknown_build_type_raises(self):
        with pytest.raises(ValueError, match="Invalid Build Type"):
            BuildFactory.create("ant", "/nonexistent")

    def test_create_forwards_target_module(self, tmp_path):
        project_root = _multimodule_project(tmp_path)
        builder = BuildFactory.create("maven", str(project_root), target_module="web")
        assert builder.target_module == "web"


class TestVendoredApiSurface:
    """Every attribute sakura consumes from javabuild must exist unchanged."""

    CONSUMED_METHODS = [
        "compile_tests_proc",
        "compile_tests",
        "run_tests_proc",
        "run_tests",
        "install_selected_projects_skip_tests_proc",
        "install_selected_projects_skip_tests",
        "add_code_coverage_dependencies",
        "add_wca_test_dependencies",
        "add_mutation_analysis_dependencies",
        "is_multi_module_project",
        "get_modules",
        "get_parent_module_path",
        "get_java_version",
        "find_compile_errors",
        "get_class_names_with_compile_errors",
    ]

    def test_consumed_methods_exist(self):
        for name in self.CONSUMED_METHODS:
            assert callable(getattr(MavenBuild, name, None)), (
                f"MavenBuild.{name} missing"
            )

    def test_init_attributes(self, tmp_path):
        project_root = _pom_project(tmp_path)
        builder = MavenBuild(str(project_root))
        assert builder.project_root == project_root
        assert builder.build_file == project_root.joinpath("pom.xml")
        assert builder.build_file_name == "pom.xml"
        assert builder.target_module is None
        assert builder.is_spring_project is False
        assert builder.build_system == "Maven"

    def test_default_maven_options_pinned(self):
        # sakura's compilation flow depends on lint/enforcer suppression
        assert set(MavenBuild.DEFAULT_MAVEN_OPTIONS) == {
            "-Drat.skip",
            "-Dfindbugs.skip",
            "-Dcheckstyle.skip",
            "-Dpmd.skip=true",
            "-Dspotbugs.skip",
            "-Denforcer.skip",
            "-Dlicense.skip=true",
        }

    def test_signature_defaults_consumed_by_sakura(self):
        compile_params = inspect.signature(MavenBuild.compile_tests_proc).parameters
        assert compile_params["pre_compile_build"].default is False
        assert compile_params["also_make"].default is True
        assert compile_params["timeout"].default == 600

        run_params = inspect.signature(MavenBuild.run_tests_proc).parameters
        assert run_params["target_tests"].default is None
        assert run_params["build_file"].default is None
        assert run_params["also_make"].default is True
        assert run_params["timeout"].default == 600

        run_tests_params = inspect.signature(MavenBuild.run_tests).parameters
        assert set(run_tests_params) >= {"target_tests", "build_file"}

        cov_params = inspect.signature(
            MavenBuild.add_code_coverage_dependencies
        ).parameters
        assert cov_params["add_java_agent"].default is False

    def test_sakura_wrappers_subclass_vendored_maven_build(self, tmp_path):
        project_root = _multimodule_project(tmp_path)
        assert issubclass(JavaMavenCompilation, MavenBuild)
        assert issubclass(JavaMavenExecution, MavenBuild)

        compilation = JavaMavenCompilation(
            project_root, module_root=project_root.joinpath("web")
        )
        assert compilation.target_module == "web"
        assert compilation.module_root == project_root.joinpath("web").resolve()

        execution = JavaMavenExecution(project_root)
        assert execution.target_module is None
        assert execution.module_root == project_root.resolve()

    def test_compilation_wrapper_parses_vendored_maven_output(self, tmp_path):
        """The subclass must keep parsing MavenBuild-style [ERROR] output."""
        project_root = _pom_project(tmp_path)
        compilation = JavaMavenCompilation(project_root)

        output = (
            "[INFO] Scanning for projects...\n"
            "[ERROR] /demo/src/test/java/com/example/FooTest.java:[12,5] ';' expected\n"
            "[ERROR]   symbol:   variable bar\n"
            "[ERROR]   location: class FooTest\n"
            "[ERROR] /demo/src/test/java/com/example/FooTest.java:[12,5] ';' expected\n"
            "[ERROR] /demo/src/test/java/com/example/FooTest.java:[40] cannot find symbol\n"
            "[INFO] BUILD FAILURE\n"
        )
        errors = compilation.parse_compilation_errors(output)

        assert len(errors) == 2  # duplicate header deduplicated
        first = errors[0]
        assert first.line == 12
        assert first.column == 5
        assert first.message == "';' expected"
        assert "symbol:   variable bar" in first.details
        assert errors[1].line == 40
        assert errors[1].column is None

    def test_runtime_method_parser_preserves_text_and_flushes_at_eof(self, tmp_path):
        project_root = _pom_project(tmp_path)
        builder = MavenBuild(str(project_root))
        error_line = "[ERROR]   FooTest.testMethod:42 Runtime failure"

        errors = builder.find_runtime_error_for_method(
            error_line, "FooTest", "testMethod"
        )

        assert errors == {42: error_line}

    def test_runtime_class_parser_flushes_unterminated_error_at_eof(self, tmp_path):
        project_root = _pom_project(tmp_path)
        builder = MavenBuild(str(project_root))
        error_line = "[ERROR]   FooTest.testMethod:42 Runtime failure"

        errors = builder.find_runtime_error_for_class(error_line, "FooTest")

        assert errors == {"testMethod": {42: error_line}}


class TestMavenCommandConstruction:
    def test_compile_tests_pre_compile_build_command(self, tmp_path, captured_maven):
        project_root = _pom_project(tmp_path)
        builder = MavenBuild(str(project_root))
        proc = builder.compile_tests_proc(pre_compile_build=True)

        assert proc.returncode == 0
        command = captured_maven[0]["command"]
        assert command[0] == MavenBuild.MAVEN_CMD
        assert command[1] == "-f"
        assert command[2].endswith("pom.xml")
        assert "clean" in command
        assert "test-compile" in command
        assert "compiler:testCompile" not in command
        for option in MavenBuild.DEFAULT_MAVEN_OPTIONS:
            assert option in command
        assert "-Dstyle.color=never" in command
        assert "-Dmaven.compiler.compilerArgs=-Xmaxerrs 20000" in command
        assert "--projects" not in command
        assert captured_maven[0]["kwargs"]["timeout"] == 600

    def test_compile_tests_incremental_command(self, tmp_path, captured_maven):
        project_root = _pom_project(tmp_path)
        builder = MavenBuild(str(project_root))
        builder.compile_tests_proc(pre_compile_build=False)

        command = captured_maven[0]["command"]
        assert "compile" in command
        assert "compiler:testCompile" in command
        assert "test-compile" not in command

    def test_module_scoped_compile_with_also_make(self, tmp_path, captured_maven):
        project_root = _multimodule_project(tmp_path)
        builder = MavenBuild(str(project_root), target_module="web")
        builder.compile_tests_proc(pre_compile_build=True, also_make=True)

        command = captured_maven[0]["command"]
        assert command[command.index("--projects") + 1] == "web"
        assert "--also-make" in command

    def test_module_scoped_compile_without_also_make(self, tmp_path, captured_maven):
        """compile_scope relies on also_make=False to isolate module errors."""
        project_root = _multimodule_project(tmp_path)
        builder = MavenBuild(str(project_root), target_module="web")
        builder.compile_tests_proc(pre_compile_build=True, also_make=False)

        command = captured_maven[0]["command"]
        assert "--projects" in command
        assert "--also-make" not in command

    def test_run_tests_with_target_selector(self, tmp_path, captured_maven):
        project_root = _pom_project(tmp_path)
        builder = MavenBuild(str(project_root))
        builder.run_tests_proc(target_tests="com.example.FooTest#testBar")

        command = captured_maven[0]["command"]
        assert "-Dtest=com.example.FooTest#testBar" in command
        assert "clean" in command
        assert "test" in command
        assert "-Dmaven.test.failure.ignore=true" in command
        assert "-DskipTests=false" in command
        assert "-Dmaven.test.skip=false" in command

    def test_run_tests_without_selector_runs_everything(self, tmp_path, captured_maven):
        project_root = _pom_project(tmp_path)
        builder = MavenBuild(str(project_root))
        builder.run_tests_proc()

        command = captured_maven[0]["command"]
        assert not any(part.startswith("-Dtest=") for part in command)

    def test_run_tests_with_custom_build_file(self, tmp_path, captured_maven):
        """IndividualTestCoverage runs tests against the mutated pom_cov.xml."""
        project_root = _pom_project(tmp_path)
        builder = MavenBuild(str(project_root))
        cov_pom = str(project_root.joinpath("pom_cov.xml"))
        result = builder.run_tests(target_tests="FooTest", build_file=cov_pom)

        assert isinstance(result, str)
        command = captured_maven[0]["command"]
        assert command[command.index("-f") + 1] == cov_pom

    def test_install_skip_tests_command(self, tmp_path, captured_maven):
        project_root = _multimodule_project(tmp_path)
        builder = MavenBuild(str(project_root), target_module="web")
        builder.install_selected_projects_skip_tests_proc()

        command = captured_maven[0]["command"]
        assert "install" in command
        assert "-Dmaven.test.skip=true" in command
        assert "-DskipTests=true" in command
        assert "-DskipITs=true" in command
        assert command[command.index("--projects") + 1] == "web"
        assert "--also-make" in command

    def test_timeout_produces_synthetic_failure(self, tmp_path, monkeypatch):
        """compile_scope depends on timeout mapping to returncode -1."""
        project_root = _pom_project(tmp_path)
        builder = MavenBuild(str(project_root))

        def timing_out_run(command, **kwargs):
            raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs["timeout"])

        monkeypatch.setattr(
            "sakura.javabuild.maven_build.subprocess.run", timing_out_run
        )
        proc = builder.compile_tests_proc(pre_compile_build=True, timeout=1)

        assert proc.returncode == -1
        assert "timed out after 1 seconds" in proc.stdout


class TestPomIntrospection:
    def test_single_module_project(self, tmp_path):
        project_root = _pom_project(tmp_path)
        builder = MavenBuild(str(project_root))
        assert builder.is_multi_module_project() is False
        assert builder.get_modules() == {}
        assert builder.get_parent_module_path() is None

    def test_plain_pom_without_namespace(self, tmp_path):
        project_root = _pom_project(tmp_path, PLAIN_POM)
        builder = MavenBuild(str(project_root))
        assert builder.is_multi_module_project() is False
        assert builder.get_java_version() == constants.MAVEN_DEFAULT_VERSION

    def test_multi_module_hierarchy(self, tmp_path):
        project_root = _multimodule_project(tmp_path)
        builder = MavenBuild(str(project_root))
        assert builder.is_multi_module_project() is True
        assert builder.get_modules() == {"web": {"web-api": {}}, "core": {}}

    def test_get_parent_module_path_uses_default_relative_path_with_custom_name(
        self, tmp_path: Path
    ) -> None:
        parent_root = tmp_path.joinpath("parent")
        child_root = parent_root.joinpath("child")
        _write_pom(parent_root, _parent_pom("child"))
        _write_pom(child_root, _child_pom("Custom Display Name"))

        builder = MavenBuild(str(child_root))

        assert builder.get_parent_module_path() == parent_root

    def test_get_parent_module_path_resolves_nested_explicit_relative_path(
        self, tmp_path: Path
    ) -> None:
        parent_root = tmp_path.joinpath("parent")
        child_root = parent_root.joinpath("services", "api")
        _write_pom(parent_root, _parent_pom("services/./api"))
        _write_pom(
            child_root,
            _child_pom("API Display Name", relative_path="../../pom.xml"),
        )

        builder = MavenBuild(str(child_root))

        assert builder.get_parent_module_path() == parent_root

    def test_get_parent_module_path_resolves_directory_relative_path(
        self, tmp_path: Path
    ) -> None:
        parent_root = tmp_path.joinpath("parent")
        child_root = parent_root.joinpath("child")
        _write_pom(parent_root, _parent_pom("child"))
        _write_pom(child_root, _child_pom("child", relative_path=".."))

        builder = MavenBuild(str(child_root))

        assert builder.get_parent_module_path() == parent_root

    def test_get_parent_module_path_returns_none_when_parent_omits_module(
        self, tmp_path: Path
    ) -> None:
        parent_root = tmp_path.joinpath("parent")
        child_root = parent_root.joinpath("child")
        _write_pom(parent_root, _parent_pom("another-child"))
        _write_pom(child_root, _child_pom("child"))

        builder = MavenBuild(str(child_root))

        assert builder.get_parent_module_path() is None

    def test_get_parent_module_path_ignores_empty_relative_path(
        self, tmp_path: Path
    ) -> None:
        parent_root = tmp_path.joinpath("parent")
        child_root = parent_root.joinpath("child")
        _write_pom(parent_root, _parent_pom("child"))
        _write_pom(child_root, _child_pom("child", relative_path=""))

        builder = MavenBuild(str(child_root))

        assert builder.get_parent_module_path() is None

    def test_java_version_from_compiler_target_property(self, tmp_path):
        project_root = _pom_project(tmp_path)
        builder = MavenBuild(str(project_root))
        assert builder.get_java_version() == "17"

    def test_java_version_strips_legacy_prefix(self, tmp_path):
        pom = NAMESPACED_POM.replace(
            "<maven.compiler.target>17</maven.compiler.target>",
            "<maven.compiler.target>1.8</maven.compiler.target>",
        )
        project_root = _pom_project(tmp_path, pom)
        builder = MavenBuild(str(project_root))
        assert builder.get_java_version() == "8"

    def test_java_version_from_java_version_property(self, tmp_path):
        pom = NAMESPACED_POM.replace(
            "<maven.compiler.target>17</maven.compiler.target>",
            "<java.version>11</java.version>",
        )
        project_root = _pom_project(tmp_path, pom)
        builder = MavenBuild(str(project_root))
        assert builder.get_java_version() == "11"

    def test_java_version_from_compiler_source_property(self, tmp_path: Path) -> None:
        project_root = _pom_project(
            tmp_path,
            _compiler_pom(
                properties="<maven.compiler.source>1.8</maven.compiler.source>"
            ),
        )

        assert MavenBuild(str(project_root)).get_java_version() == "8"

    def test_java_version_from_compiler_plugin_config(self, tmp_path):
        pom = NAMESPACED_POM.replace(
            """  <properties>
    <maven.compiler.target>17</maven.compiler.target>
  </properties>
""",
            """  <build>
    <plugins>
      <plugin>
        <artifactId>maven-compiler-plugin</artifactId>
        <configuration>
          <release>21</release>
        </configuration>
      </plugin>
    </plugins>
  </build>
""",
        )
        project_root = _pom_project(tmp_path, pom)
        builder = MavenBuild(str(project_root))
        assert builder.get_java_version() == "21"

    def test_java_version_from_compiler_plugin_source(self, tmp_path: Path) -> None:
        project_root = _pom_project(
            tmp_path, _compiler_pom(configuration="<source>11</source>")
        )

        assert MavenBuild(str(project_root)).get_java_version() == "11"

    @pytest.mark.parametrize(
        ("properties", "expected"),
        [
            (
                """<maven.compiler.target>17</maven.compiler.target>
<maven.compiler.release>21</maven.compiler.release>
<maven.compiler.source>11</maven.compiler.source>""",
                "17",
            ),
            (
                """<maven.compiler.release>21</maven.compiler.release>
<maven.compiler.source>11</maven.compiler.source>""",
                "21",
            ),
            ("<maven.compiler.source>11</maven.compiler.source>", "11"),
        ],
    )
    def test_java_version_property_precedence_over_plugin(
        self, tmp_path: Path, properties: str, expected: str
    ) -> None:
        project_root = _pom_project(
            tmp_path,
            _compiler_pom(
                properties=properties,
                configuration="<target>22</target><release>23</release><source>24</source>",
            ),
        )

        assert MavenBuild(str(project_root)).get_java_version() == expected

    @pytest.mark.parametrize(
        ("configuration", "expected"),
        [
            (
                "<target>17</target><release>21</release><source>11</source>",
                "17",
            ),
            ("<release>21</release><source>11</source>", "21"),
        ],
    )
    def test_java_version_plugin_precedence(
        self, tmp_path: Path, configuration: str, expected: str
    ) -> None:
        project_root = _pom_project(
            tmp_path, _compiler_pom(configuration=configuration)
        )

        assert MavenBuild(str(project_root)).get_java_version() == expected


class TestWcaPomMutation:
    @pytest.mark.parametrize(
        ("include_spring", "expected_spring_artifacts"),
        [
            (False, set()),
            (True, {"spring-test", "spring-web"}),
        ],
    )
    def test_spring_dependencies_follow_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        include_spring: bool,
        expected_spring_artifacts: set[str],
    ) -> None:
        dependencies = copy.deepcopy(
            MavenBuild.WCA_TESTGEN_MAVEN_DEPENDENCIES_WITH_SPRING
        )
        monkeypatch.setattr(
            MavenBuild, "WCA_TESTGEN_MAVEN_DEPENDENCIES_WITH_SPRING", dependencies
        )
        project_root = _pom_project(tmp_path, PLAIN_POM)

        MavenBuild(str(project_root)).add_wca_test_dependencies(
            is_add_spring_dependency=include_spring
        )

        root = ET.parse(project_root.joinpath("pom.xml")).getroot()
        spring_artifacts = {
            _child_text(dependency, "artifactId")
            for dependency in _elements(root, "dependency")
            if _child_text(dependency, "groupId") == "org.springframework"
        }
        assert spring_artifacts == expected_spring_artifacts

    def test_repeated_spring_dependency_additions_do_not_mutate_configuration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected_dependencies = copy.deepcopy(
            MavenBuild.WCA_TESTGEN_MAVEN_DEPENDENCIES_WITH_SPRING
        )
        dependencies = copy.deepcopy(expected_dependencies)
        monkeypatch.setattr(
            MavenBuild, "WCA_TESTGEN_MAVEN_DEPENDENCIES_WITH_SPRING", dependencies
        )

        for call_number in range(2):
            project_root = _pom_project(
                tmp_path.joinpath(f"call-{call_number}"), PLAIN_POM
            )
            MavenBuild(str(project_root)).add_wca_test_dependencies(
                is_add_spring_dependency=True
            )

            root = ET.parse(project_root.joinpath("pom.xml")).getroot()
            spring_artifacts = {
                _child_text(dependency, "artifactId")
                for dependency in _elements(root, "dependency")
                if _child_text(dependency, "groupId") == "org.springframework"
            }
            assert spring_artifacts == {"spring-test", "spring-web"}

        assert dependencies == expected_dependencies

    def test_namespaced_mutation_is_idempotent_on_same_builder(
        self, tmp_path: Path
    ) -> None:
        project_root = _pom_project(tmp_path)
        builder = MavenBuild(str(project_root))

        builder.add_wca_test_dependencies()
        builder.add_wca_test_dependencies()

        root = ET.parse(project_root.joinpath("pom.xml")).getroot()
        dependency_sections = [
            child for child in root if _local_name(child.tag) == "dependencies"
        ]
        artifact_ids = [
            _child_text(dependency, "artifactId")
            for dependency in _elements(root, "dependency")
        ]
        assert len(dependency_sections) == 1
        assert len(artifact_ids) == 3
        assert set(artifact_ids) == {
            "junit-jupiter-api",
            "mockito-core",
            "mockito-junit-jupiter",
        }


class TestCoveragePomMutation:
    def _mutated_pom_root(self, tmp_path, body=NAMESPACED_POM, add_java_agent=True):
        project_root = _pom_project(tmp_path, body)
        builder = MavenBuild(str(project_root))
        output_pom = project_root.joinpath("pom_cov.xml")
        builder.add_code_coverage_dependencies(
            output_build_file=str(output_pom), add_java_agent=add_java_agent
        )
        assert output_pom.exists()
        return ET.parse(output_pom).getroot()

    def test_jacoco_plugin_injected_with_pinned_version(self, tmp_path):
        root = self._mutated_pom_root(tmp_path)
        plugins = _find_plugin(root, "jacoco-maven-plugin")
        assert len(plugins) == 1

        plugin = plugins[0]
        assert _child_text(plugin, "groupId") == "org.jacoco"
        assert _child_text(plugin, "version") == constants.JACOCO_VERSION

        goals = {goal.text for goal in _elements(plugin, "goal")}
        assert goals == {"prepare-agent", "report"}

        config_texts = {
            _local_name(e.tag): (e.text or "") for e in _elements(plugin, "destFile")
        }
        assert config_texts.get("destFile") == constants.MAVEN_JACOCO_COV_FILE

    def test_java_agent_dependency_and_surefire_argline(self, tmp_path):
        root = self._mutated_pom_root(tmp_path, add_java_agent=True)

        agent_deps = [
            dep
            for dep in _elements(root, "dependency")
            if _child_text(dep, "artifactId") == "org.jacoco.agent"
        ]
        assert len(agent_deps) == 1
        agent = agent_deps[0]
        assert _child_text(agent, "version") == constants.JACOCO_VERSION
        assert _child_text(agent, "classifier") == "runtime"
        assert _child_text(agent, "scope") == "test"

        surefire = _find_plugin(root, "maven-surefire-plugin")
        assert len(surefire) == 1
        arglines = _elements(surefire[0], "argLine")
        assert len(arglines) == 1
        argline = arglines[0].text
        assert "-javaagent:" in argline
        assert f"org.jacoco.agent-{constants.JACOCO_VERSION}-runtime.jar" in argline
        assert "output=none,jmx=true" in argline

    def test_no_agent_when_flag_disabled(self, tmp_path):
        root = self._mutated_pom_root(tmp_path, add_java_agent=False)
        agent_deps = [
            dep
            for dep in _elements(root, "dependency")
            if _child_text(dep, "artifactId") == "org.jacoco.agent"
        ]
        assert agent_deps == []
        assert _find_plugin(root, "maven-surefire-plugin") == []

    def test_existing_jacoco_plugin_not_duplicated(self, tmp_path):
        pom_with_jacoco = NAMESPACED_POM.replace(
            """  <properties>
    <maven.compiler.target>17</maven.compiler.target>
  </properties>
""",
            """  <build>
    <plugins>
      <plugin>
        <groupId>org.jacoco</groupId>
        <artifactId>jacoco-maven-plugin</artifactId>
        <version>0.8.14</version>
      </plugin>
    </plugins>
  </build>
""",
        )
        root = self._mutated_pom_root(tmp_path, body=pom_with_jacoco)
        plugins = _find_plugin(root, "jacoco-maven-plugin")
        assert len(plugins) == 1  # pre-existing plugin kept, none added
        assert _child_text(plugins[0], "version") == "0.8.14"

    def test_plain_pom_mutation_works_without_namespace(self, tmp_path):
        root = self._mutated_pom_root(tmp_path, body=PLAIN_POM)
        assert len(_find_plugin(root, "jacoco-maven-plugin")) == 1

    def test_namespaced_mutation_is_idempotent_on_same_builder(
        self, tmp_path: Path
    ) -> None:
        project_root = _pom_project(tmp_path)
        output_pom = project_root.joinpath("pom_cov.xml")
        builder = MavenBuild(str(project_root))

        for _ in range(2):
            builder.add_code_coverage_dependencies(
                output_build_file=str(output_pom), add_java_agent=True
            )

        root = ET.parse(output_pom).getroot()
        build_sections = [child for child in root if _local_name(child.tag) == "build"]
        dependency_sections = [
            child for child in root if _local_name(child.tag) == "dependencies"
        ]
        plugins_sections = [
            child for child in build_sections[0] if _local_name(child.tag) == "plugins"
        ]
        agent_dependencies = [
            dependency
            for dependency in _elements(root, "dependency")
            if _child_text(dependency, "artifactId") == "org.jacoco.agent"
        ]
        surefire = _find_plugin(root, "maven-surefire-plugin")

        assert len(build_sections) == 1
        assert len(dependency_sections) == 1
        assert len(plugins_sections) == 1
        assert len(_find_plugin(root, "jacoco-maven-plugin")) == 1
        assert len(agent_dependencies) == 1
        assert len(surefire) == 1
        arg_line = _elements(surefire[0], "argLine")[0].text or ""
        assert arg_line.count("-javaagent:") == 1

    def test_existing_surefire_plugin_is_reused_and_configuration_preserved(
        self, tmp_path: Path
    ) -> None:
        pom_with_surefire = NAMESPACED_POM.replace(
            """  <properties>
    <maven.compiler.target>17</maven.compiler.target>
  </properties>
""",
            """  <build>
    <plugins>
      <plugin>
        <artifactId>maven-surefire-plugin</artifactId>
        <version>3.5.2</version>
        <configuration>
          <reuseForks>false</reuseForks>
          <argLine>-Xmx512m</argLine>
        </configuration>
      </plugin>
    </plugins>
  </build>
""",
        )
        project_root = _pom_project(tmp_path, pom_with_surefire)
        output_pom = project_root.joinpath("pom_cov.xml")
        builder = MavenBuild(str(project_root))

        for _ in range(2):
            builder.add_code_coverage_dependencies(
                output_build_file=str(output_pom), add_java_agent=True
            )

        root = ET.parse(output_pom).getroot()
        surefire = _find_plugin(root, "maven-surefire-plugin")
        assert len(surefire) == 1
        assert _child_text(surefire[0], "groupId") is None
        assert _child_text(surefire[0], "version") == "3.5.2"
        assert (
            _child_text(_elements(surefire[0], "configuration")[0], "reuseForks")
            == "false"
        )
        arg_line = _elements(surefire[0], "argLine")[0].text or ""
        assert arg_line.startswith("-Xmx512m ")
        assert arg_line.count("-javaagent:") == 1


class TestMutationAnalysisPomMutation:
    def test_namespaced_mutation_is_idempotent_on_same_builder(
        self, tmp_path: Path
    ) -> None:
        project_root = _pom_project(tmp_path)
        output_pom = project_root.joinpath("pom_pitest.xml")
        builder = MavenBuild(str(project_root))

        for _ in range(2):
            builder.add_mutation_analysis_dependencies(
                output_build_file=str(output_pom),
                target_tests=["com.example.*Test"],
                excluded_tests=["com.example.SlowTest"],
                target_classes=["com.example.*"],
            )

        root = ET.parse(output_pom).getroot()
        build_sections = [child for child in root if _local_name(child.tag) == "build"]
        plugins_sections = [
            child for child in build_sections[0] if _local_name(child.tag) == "plugins"
        ]

        assert len(build_sections) == 1
        assert len(plugins_sections) == 1
        assert len(_find_plugin(root, "pitest-maven")) == 1
        assert len(_elements(root, "targetTests")) == 1
        assert len(_elements(root, "excludedTestClasses")) == 1
        assert len(_elements(root, "targetClasses")) == 1


class TestReasterTestCodeContract:
    """The vendored TEST_CODE template must match what
    IndividualTestCoverage generates, injects, and later parses."""

    def test_package_placeholder_renders(self):
        assert "package <package_name>;" in TEST_CODE
        rendered = TEST_CODE.replace("<package_name>", "com.example.app")
        assert "package com.example.app;" in rendered
        assert "<package_name>" not in rendered

    def test_watcher_class_name_matches_coverage_constant(self):
        assert f"public class {TEST_WATCHER_CLASS_NAME} " in TEST_CODE

    def test_base_dir_matches_jacoco_test_folder_constant(self):
        assert f'BASE_DIR = "{JACOCO_TEST_FOLDER}"' in TEST_CODE

    def test_junit5_extension_callbacks_present(self):
        assert "org.junit.jupiter.api.extension" in TEST_CODE
        assert "BeforeTestExecutionCallback" in TEST_CODE
        assert "AfterTestExecutionCallback" in TEST_CODE

    def test_jacoco_runtime_agent_used(self):
        assert "org.jacoco.agent.rt.RT" in TEST_CODE
        assert "getExecutionData" in TEST_CODE

    def test_exec_file_naming_matches_collector_parsing(self):
        # IndividualTestCoverage parses "<class>__<method>.exec" (double
        # underscore); the watcher must produce exactly that shape.
        assert '"__" +' in TEST_CODE
        assert 'testName + ".exec"' in TEST_CODE

    def test_jacococli_jar_exists_and_matches_pinned_version(self):
        assert JACOCO_CLI_JAR.exists(), f"missing {JACOCO_CLI_JAR}"
        assert constants.JACOCO_VERSION in JACOCO_CLI_JAR.name


class TestEndToEndPetclinicIndividualCoverage:
    """Drives the complete migrated pipeline with real Maven runs:
    vendored MavenBuild mutates the POM, the vendored TEST_CODE watcher is
    injected, tests execute with the JaCoCo agent, and per-test coverage
    is collected via jacococli."""

    def test_individual_test_coverage_end_to_end(self, petclinic_paths):
        project_root = petclinic_paths.project_root
        test_class = "org.springframework.samples.petclinic.model.ValidatorTests"
        test_method = "shouldNotValidateWhenFirstNameEmpty"

        test_file = project_root.joinpath(
            "src/test/java/org/springframework/samples/petclinic/model/ValidatorTests.java"
        )
        original_content = test_file.read_text()
        watcher_file = project_root.joinpath(
            "src/test/java/org/springframework/samples/petclinic",
            f"{TEST_WATCHER_CLASS_NAME}.java",
        )

        try:
            coverage_runner = IndividualTestCoverage(project_root=project_root)
            # petclinic runs spring-javaformat validation, which rejects the
            # injected watcher source; skip it like the compilation wrappers do
            coverage_runner.builder.is_spring_project = True
            assert (
                coverage_runner.package_root == "org.springframework.samples.petclinic"
            )

            coverage = coverage_runner.generate(
                tests_to_run=[(test_class, test_method)]
            )

            content_after_run = test_file.read_text()
            watcher_exists_after_run = watcher_file.exists()
        finally:
            # Keep the petclinic submodule pristine even on failure
            test_file.write_text(original_content)
            watcher_file.unlink(missing_ok=True)
            project_root.joinpath("pom_cov.xml").unlink(missing_ok=True)
            shutil.rmtree(project_root.joinpath(JACOCO_TEST_FOLDER), ignore_errors=True)

        # generate() must clean up after itself
        assert content_after_run == original_content
        assert not watcher_exists_after_run

        # Per-test coverage must have been collected for the targeted test
        assert test_class in coverage
        entries = coverage[test_class]
        entry = next(e for e in entries if e["test_name"] == test_method)
        details = entry["coverage_details"]
        assert details, "expected non-empty per-test coverage details"

        # The validator test exercises Person setters, so Person must show
        # covered lines in the per-test report.
        person = "org.springframework.samples.petclinic.model.Person"
        assert person in details
        assert details[person]["covered_lines"], "Person should have covered lines"
        for class_coverage in details.values():
            assert all(
                isinstance(line, int) for line in class_coverage["covered_lines"]
            )
