from __future__ import annotations

import os
from pathlib import Path

from sakura.utils.file_io.pom_processor import PomProcessor


def _dependency_keys(
    module_root: Path, parent_roots: list[Path] | None = None
) -> set[tuple[str, str]]:
    dependencies = PomProcessor.identify_dependencies(
        module_root=module_root, parent_roots=parent_roots
    )
    return {(dep.group_id, dep.artifact_id) for dep in dependencies}


def _assert_dependencies_present(
    actual: set[tuple[str, str]], expected: set[tuple[str, str]]
) -> None:
    missing = expected - actual
    assert not missing, f"Missing dependencies: {sorted(missing)}"


def test_identify_dependencies_includes_parent_pom() -> None:
    original_cwd = os.getcwd()
    try:
        os.chdir(Path(__file__).resolve().parent)
        project_root = Path("../resources/datasets/commons-numbers")
        module_root = project_root / "commons-numbers-core"
        dependency_keys = _dependency_keys(
            module_root=module_root,
            parent_roots=[project_root],
        )
    finally:
        os.chdir(original_cwd)

    expected = {
        ("org.junit.jupiter", "junit-jupiter"),
        ("org.apache.commons", "commons-rng-sampling"),
        ("org.apache.commons", "commons-rng-simple"),
    }
    _assert_dependencies_present(dependency_keys, expected)


def test_identify_dependencies_single_module() -> None:
    original_cwd = os.getcwd()
    try:
        os.chdir(Path(__file__).resolve().parent)
        project_root = Path("resources/spring-petclinic")
        dependency_keys = _dependency_keys(module_root=project_root)
    finally:
        os.chdir(original_cwd)

    expected = {
        ("org.springframework.boot", "spring-boot-starter-webmvc"),
        ("jakarta.xml.bind", "jakarta.xml.bind-api"),
    }
    _assert_dependencies_present(dependency_keys, expected)
