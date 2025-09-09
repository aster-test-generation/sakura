from __future__ import annotations

from pathlib import Path
from typing import List
import xml.etree.ElementTree as ET
from pydantic import BaseModel
from nltest.utils.exceptions import PomXmlNotFoundError


class MavenDependency(BaseModel):
    """Represents a Maven dependency coordinate."""

    group_id: str
    artifact_id: str


class PomProcessor:
    @staticmethod
    def identify_dependencies(project_root: Path) -> List[MavenDependency]:
        """
        Identify direct Maven dependencies from the top-level section.
        Raises PomXmlNotFoundError if the root pom.xml is missing.
        Returns an empty list if the POM exists but cannot be parsed.
        """

        # Resolve the pom.xml path
        pom_path = project_root / "pom.xml"

        if not pom_path.exists() or not pom_path.is_file():
            raise PomXmlNotFoundError(
                "Root pom.xml not found.", extra_info={"pom_path": str(pom_path)}
            )

        try:
            root = ET.parse(pom_path).getroot()
        except ET.ParseError:
            return []

        namespace = ""
        if root.tag.startswith("{") and "}" in root.tag:
            namespace = root.tag[1 : root.tag.find("}")]

        def q(tag: str) -> str:
            return f"{{{namespace}}}{tag}" if namespace else tag

        dependencies: List[MavenDependency] = []

        # Only consider <dependencies> directly under the root <project>
        deps_container = root.find(q("dependencies"))
        if deps_container is None:
            return []

        for dep_el in deps_container.findall(q("dependency")):
            group_el = dep_el.find(q("groupId"))
            artifact_el = dep_el.find(q("artifactId"))

            if group_el is None or artifact_el is None:
                continue

            group_id = (group_el.text or "").strip()
            artifact_id = (artifact_el.text or "").strip()

            if group_id and artifact_id:
                dependencies.append(
                    MavenDependency(group_id=group_id, artifact_id=artifact_id)
                )

        return dependencies
