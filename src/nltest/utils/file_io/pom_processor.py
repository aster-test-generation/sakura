from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import re
import xml.etree.ElementTree as ET

from pydantic import BaseModel

from nltest.utils.exceptions import PomXmlNotFoundError


class MavenDependency(BaseModel):
    """Represents a Maven dependency coordinate."""

    group_id: str
    artifact_id: str


class PomProcessor:
    @staticmethod
    def identify_dependencies(
        module_root: Path,
        parent_roots: Sequence[Path] | None = None,
    ) -> List[MavenDependency]:
        """
        Identify Maven dependencies for a module, including inherited parent POMs.
        Raises PomXmlNotFoundError if the module pom.xml is missing.
        Returns an empty list if the module POM exists but cannot be parsed.
        """

        module_root = Path(module_root)
        parent_root_paths = [Path(root) for root in parent_roots or []]

        module_pom = module_root / "pom.xml"
        if not module_pom.exists() or not module_pom.is_file():
            raise PomXmlNotFoundError(
                "Module pom.xml not found.",
                extra_info={"pom_path": str(module_pom)},
            )

        pom_chain: List[Tuple[Path, ET.Element, str]] = []
        visited: set[Path] = set()
        property_pattern = re.compile(r"\$\{([^}]+)\}")

        def parse_pom(pom_path: Path) -> Tuple[ET.Element, str] | None:
            try:
                root = ET.parse(pom_path).getroot()
            except ET.ParseError:
                return None

            namespace = ""
            if root.tag.startswith("{") and "}" in root.tag:
                namespace = root.tag[1 : root.tag.find("}")]
            return root, namespace

        def qualify(namespace: str, tag: str) -> str:
            return f"{{{namespace}}}{tag}" if namespace else tag

        def get_text(element: ET.Element | None, tag: str, namespace: str) -> str:
            if element is None:
                return ""
            child = element.find(qualify(namespace, tag))
            if child is None or child.text is None:
                return ""
            return child.text.strip()

        def extract_properties(root: ET.Element, namespace: str) -> Dict[str, str]:
            properties_el = root.find(qualify(namespace, "properties"))
            if properties_el is None:
                return {}

            properties: Dict[str, str] = {}
            for prop in list(properties_el):
                key = prop.tag
                if key.startswith("{") and "}" in key:
                    key = key[key.find("}") + 1 :]
                value = (prop.text or "").strip()
                if key and value:
                    properties[key] = value
            return properties

        def resolve_placeholders(value: str, properties: Dict[str, str]) -> str:
            if not value:
                return ""

            def replace(match: re.Match[str]) -> str:
                key = match.group(1)
                return properties.get(key, match.group(0))

            return property_pattern.sub(replace, value)

        def apply_standard_properties(
            properties: Dict[str, str],
            *,
            group_id: str,
            artifact_id: str,
            version: str,
            parent_group_id: str,
            parent_artifact_id: str,
            parent_version: str,
        ) -> Dict[str, str]:
            updated = dict(properties)
            if group_id:
                updated["project.groupId"] = group_id
                updated["pom.groupId"] = group_id
            if artifact_id:
                updated["project.artifactId"] = artifact_id
                updated["pom.artifactId"] = artifact_id
            if version:
                updated["project.version"] = version
                updated["pom.version"] = version
            if parent_group_id:
                updated["project.parent.groupId"] = parent_group_id
                updated["parent.groupId"] = parent_group_id
            if parent_artifact_id:
                updated["project.parent.artifactId"] = parent_artifact_id
                updated["parent.artifactId"] = parent_artifact_id
            if parent_version:
                updated["project.parent.version"] = parent_version
                updated["parent.version"] = parent_version
            return updated

        def extract_dependencies(
            root: ET.Element,
            namespace: str,
            properties: Dict[str, str],
        ) -> List[MavenDependency]:
            deps_container = root.find(qualify(namespace, "dependencies"))
            if deps_container is None:
                return []

            dependencies: List[MavenDependency] = []
            for dep_el in deps_container.findall(qualify(namespace, "dependency")):
                group_id = resolve_placeholders(
                    get_text(dep_el, "groupId", namespace), properties
                )
                artifact_id = resolve_placeholders(
                    get_text(dep_el, "artifactId", namespace), properties
                )
                if group_id and artifact_id:
                    dependencies.append(
                        MavenDependency(group_id=group_id, artifact_id=artifact_id)
                    )
            return dependencies

        def resolve_parent_pom(
            root: ET.Element,
            namespace: str,
            base_dir: Path,
        ) -> Path | None:
            parent_el = root.find(qualify(namespace, "parent"))
            if parent_el is None:
                return None

            relative_el = parent_el.find(qualify(namespace, "relativePath"))
            if relative_el is None:
                relative_path = Path("..") / "pom.xml"
            else:
                relative_text = (relative_el.text or "").strip()
                if not relative_text:
                    relative_path = None
                else:
                    relative_path = Path(relative_text)

            candidates: List[Path] = []
            if relative_path is not None:
                candidates.append(base_dir / relative_path)
            for parent_root in parent_root_paths:
                if parent_root.name == "pom.xml":
                    candidates.append(parent_root)
                else:
                    candidates.append(parent_root / "pom.xml")

            for candidate in candidates:
                if candidate.exists() and candidate.is_file():
                    return candidate
            return None

        current_pom = module_pom
        while current_pom is not None and current_pom not in visited:
            visited.add(current_pom)
            parsed = parse_pom(current_pom)
            if parsed is None:
                if current_pom == module_pom:
                    return []
                break
            root, namespace = parsed
            pom_chain.append((current_pom, root, namespace))
            current_pom = resolve_parent_pom(root, namespace, current_pom.parent)

        dependencies: List[MavenDependency] = []
        seen: set[tuple[str, str]] = set()
        properties: Dict[str, str] = {}

        for _, root, namespace in reversed(pom_chain):
            parent_el = root.find(qualify(namespace, "parent"))
            parent_group_id = get_text(parent_el, "groupId", namespace)
            parent_artifact_id = get_text(parent_el, "artifactId", namespace)
            parent_version = get_text(parent_el, "version", namespace)

            group_id = (
                get_text(root, "groupId", namespace)
                or parent_group_id
                or properties.get("project.groupId", "")
            )
            artifact_id = get_text(root, "artifactId", namespace) or properties.get(
                "project.artifactId", ""
            )
            version = (
                get_text(root, "version", namespace)
                or parent_version
                or properties.get("project.version", "")
            )

            properties = {**properties, **extract_properties(root, namespace)}
            properties = apply_standard_properties(
                properties,
                group_id=group_id,
                artifact_id=artifact_id,
                version=version,
                parent_group_id=parent_group_id,
                parent_artifact_id=parent_artifact_id,
                parent_version=parent_version,
            )

            for dependency in extract_dependencies(root, namespace, properties):
                key = (dependency.group_id, dependency.artifact_id)
                if key not in seen:
                    dependencies.append(dependency)
                    seen.add(key)

        return dependencies
