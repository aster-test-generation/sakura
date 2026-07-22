from __future__ import annotations

import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from cldk import CLDK
from cldk.analysis import AnalysisLevel
from cldk.analysis.java import JavaAnalysis

from sakura.test2nl.context import Test2NLContextBuilder
from sakura.test2nl.model.models import (
    ClassContext,
    FieldDeclaration,
    MethodContext,
    Test2NLContext,
    Test2NLEntry,
)
from sakura.utils.analysis import CommonAnalysis, Reachability


class DescriptionContextService:
    """Load project analyses and render prompt-equivalent Java context on demand."""

    def __init__(
        self,
        projects_dir: Path,
        analysis_dir: Path,
        analysis_loader: Callable[[str], JavaAnalysis] | None = None,
    ) -> None:
        self.projects_dir = projects_dir.resolve()
        self.analysis_dir = analysis_dir.resolve()
        self._analysis_loader = analysis_loader
        self._project_cache: dict[
            str, tuple[JavaAnalysis, Test2NLContextBuilder]
        ] = {}
        self._render_cache: dict[int, str] = {}
        self._lock = threading.RLock()

    def render_entry(self, entry: Test2NLEntry) -> str:
        with self._lock:
            cached = self._render_cache.get(entry.id)
            if cached is not None:
                return cached
            analysis, builder = self._load_project(entry.project_name)
            owner, signature = self._resolve_method(analysis, entry)
            context = builder.build(signature, owner)
            rendered = JavaContextRenderer.render(context)
            self._render_cache[entry.id] = rendered
            return rendered

    def _load_project(
        self, project_name: str
    ) -> tuple[JavaAnalysis, Test2NLContextBuilder]:
        cached = self._project_cache.get(project_name)
        if cached is not None:
            return cached

        if self._analysis_loader is not None:
            analysis = self._analysis_loader(project_name)
        else:
            project_path = self.projects_dir / project_name
            if not project_path.is_dir():
                raise FileNotFoundError(f"Project directory not found: {project_path}")
            project_analysis_dir = self.analysis_dir / project_name
            analysis_json = project_analysis_dir / "analysis.json"
            project_analysis_dir.mkdir(parents=True, exist_ok=True)
            analysis = CLDK(language="java").analysis(
                project_path=str(project_path),
                analysis_backend_path=None,
                analysis_level=AnalysisLevel.symbol_table,
                analysis_json_path=project_analysis_dir,
                eager=not analysis_json.exists(),
            )

        _, application_classes, test_utility_classes = CommonAnalysis(
            analysis
        ).categorize_classes()
        result = (
            analysis,
            Test2NLContextBuilder(
                analysis, application_classes, test_utility_classes
            ),
        )
        self._project_cache[project_name] = result
        return result

    @staticmethod
    def _resolve_method(
        analysis: JavaAnalysis, entry: Test2NLEntry
    ) -> tuple[str, str]:
        signatures = [entry.method_signature]
        simplified = CommonAnalysis.simplify_method_signature(entry.method_signature)
        if simplified not in signatures:
            signatures.append(simplified)
        for signature in signatures:
            if analysis.get_method(entry.qualified_class_name, signature):
                return entry.qualified_class_name, signature

        common = CommonAnalysis(analysis)
        frameworks = common.get_testing_frameworks_for_class(
            entry.qualified_class_name
        )
        reachable = Reachability(analysis).get_reachable_test_methods(
            entry.qualified_class_name, frameworks
        )
        normalized_targets = {
            CommonAnalysis.simplify_method_signature(signature).replace(" ", "")
            for signature in signatures
        }
        for owner, method_signatures in reachable.items():
            for signature in method_signatures:
                normalized = CommonAnalysis.simplify_method_signature(
                    signature
                ).replace(" ", "")
                if normalized in normalized_targets:
                    return owner, signature
        raise ValueError(
            f"Method {entry.method_signature} in {entry.qualified_class_name} "
            "could not be resolved directly or through inheritance"
        )


class JavaContextRenderer:
    """Render structured prompt context as a readable, source-like Java block."""

    @classmethod
    def render(cls, context: Test2NLContext) -> str:
        classes: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"fields": [], "methods": []}
        )
        primary = context.qualified_class_name
        classes[primary]["fields"].extend(context.field_declarations)

        for method in context.setup_methods:
            owner = method.qualified_class_name or primary
            classes[owner]["methods"].append(("Setup fixture", method, []))
        classes[primary]["methods"].append(
            ("Test method", context.test_method, context.method_annotations)
        )
        for method in context.helper_methods:
            owner = method.qualified_class_name or primary
            classes[owner]["methods"].append(("Helper method", method, []))
        for method in context.teardown_methods:
            owner = method.qualified_class_name or primary
            classes[owner]["methods"].append(("Teardown fixture", method, []))

        blocks = [
            cls._render_test_class(
                primary,
                classes.pop(primary),
                context.class_annotations,
                f"// TEST SUITE: {primary} declares the test method",
            )
        ]
        for owner, members in classes.items():
            blocks.append(
                cls._render_test_class(
                    owner,
                    members,
                    [],
                    f"// TEST SUITE: {owner} provides fixtures and helpers "
                    "referenced by the test",
                )
            )
        for application_class in context.application_classes:
            blocks.append(cls._render_application_class(application_class))
        return "\n\n".join(blocks).rstrip() + "\n"

    @classmethod
    def _render_test_class(
        cls,
        owner: str,
        members: dict[str, Any],
        annotations: list[str],
        header: str,
    ) -> str:
        lines = [header, *annotations, f"class {cls._simple_name(owner)} {{"]
        for field in members["fields"]:
            lines.extend(cls._indent(cls._render_field(field)))
        if members["fields"] and members["methods"]:
            lines.append("")
        for index, (role, method, method_annotations) in enumerate(
            members["methods"]
        ):
            if index:
                lines.append("")
            method_lines = [f"// {role}", *method_annotations]
            method_lines.extend(cls._render_method(method))
            lines.extend(cls._indent(method_lines))
        lines.append("}")
        return "\n".join(lines)

    @classmethod
    def _render_application_class(cls, context: ClassContext) -> str:
        modifiers = " ".join(context.modifiers or [])
        declaration = f"{modifiers} class {context.simple_class_name}".strip()
        if context.extends:
            declaration += " extends " + ", ".join(context.extends)
        qualified = context.qualified_class_name or context.simple_class_name
        lines = [
            f"// APPLICATION CODE: {qualified}; only the signatures below "
            "were visible during description generation",
            *(context.annotations or []),
            declaration + " {",
        ]
        for field in context.field_declarations or []:
            lines.extend(cls._indent(cls._render_field(field)))
        methods = context.relevant_class_methods or []
        if context.field_declarations and methods:
            lines.append("")
        for index, method in enumerate(methods):
            if index:
                lines.append("")
            signature = method.method_signature
            if signature.startswith("<init>"):
                signature = context.simple_class_name + signature[len("<init>") :]
            lines.extend(cls._indent([signature + ";"]))
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def _render_field(field: FieldDeclaration) -> list[str]:
        lines = list(field.annotations or [])
        declaration = " ".join(
            [
                *(field.modifiers or []),
                field.type or "Object",
                ", ".join(field.variables or ["field"]),
            ]
        )
        lines.append(declaration + ";")
        return lines

    @staticmethod
    def _render_method(method: MethodContext) -> list[str]:
        if method.code:
            return method.code.splitlines()
        return [method.method_signature + ";"]

    @staticmethod
    def _simple_name(qualified_name: str) -> str:
        return qualified_name.rsplit(".", 1)[-1].rsplit("$", 1)[-1]

    @staticmethod
    def _indent(lines: list[str], spaces: int = 4) -> list[str]:
        prefix = " " * spaces
        return [prefix + line if line else "" for line in lines]
