from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from cldk.analysis.java import JavaAnalysis
from cldk.models.java import JCallable
from cldk.models.java.models import JComment, JField, JMethodDetail, JType


class AppJavaAnalysis(JavaAnalysis):
    """
    Wrapper over CLDK's JavaAnalysis that blocks out all non-application classes from 
    retrieval, effectively making existing test suites invisible.
    """
    def __init__(
        self,
        *,
        application_classes: list[str],
        project_dir: str | Path | None,
        source_code: str | None,
        analysis_backend_path: str | None,
        analysis_json_path: str | Path | None,
        analysis_level: str,
        target_files: list[str] | None,
        eager_analysis: bool,
    ) -> None:
        super().__init__(
            project_dir=project_dir,
            source_code=source_code,
            analysis_backend_path=analysis_backend_path,
            analysis_json_path=analysis_json_path,
            analysis_level=analysis_level,
            target_files=target_files,
            eager_analysis=eager_analysis,
        )
        self._allowed_classes: set[str] = set(application_classes)
        for class_name in list(self._allowed_classes):
            if "$" in class_name:
                self._allowed_classes.add(class_name.replace("$", "."))

    def _is_allowed_class(self, qualified_class_name: str) -> bool:
        if not qualified_class_name:
            return False
        if qualified_class_name in self._allowed_classes:
            return True
        normalized = qualified_class_name.replace("$", ".")
        return normalized in self._allowed_classes

    def get_classes(self) -> Dict[str, JType]:
        classes = super().get_classes()
        return {
            qualified_name: class_info
            for qualified_name, class_info in classes.items()
            if self._is_allowed_class(qualified_name)
        }

    def get_classes_by_criteria(
        self,
        inclusions: list[str] | None = None,
        exclusions: list[str] | None = None,
    ) -> Dict[str, JType]:
        classes = super().get_classes_by_criteria(
            inclusions=inclusions,
            exclusions=exclusions,
        )
        return {
            qualified_name: class_info
            for qualified_name, class_info in classes.items()
            if self._is_allowed_class(qualified_name)
        }

    def get_methods(self) -> Dict[str, Dict[str, JCallable]]:
        methods = super().get_methods()
        return {
            qualified_name: method_map
            for qualified_name, method_map in methods.items()
            if self._is_allowed_class(qualified_name)
        }

    def get_entry_point_classes(self) -> Dict[str, JType]:
        classes = super().get_entry_point_classes()
        return {
            qualified_name: class_info
            for qualified_name, class_info in classes.items()
            if self._is_allowed_class(qualified_name)
        }

    def get_entry_point_methods(self) -> Dict[str, Dict[str, JCallable]]:
        methods = super().get_entry_point_methods()
        return {
            qualified_name: method_map
            for qualified_name, method_map in methods.items()
            if self._is_allowed_class(qualified_name)
        }

    def get_class(self, qualified_class_name: str) -> JType | None:
        if not self._is_allowed_class(qualified_class_name):
            return None
        return super().get_class(qualified_class_name)

    def get_method(
        self, qualified_class_name: str, qualified_method_name: str
    ) -> JCallable | None:
        if not self._is_allowed_class(qualified_class_name):
            return None
        return super().get_method(qualified_class_name, qualified_method_name)

    def get_method_parameters(
        self, qualified_class_name: str, qualified_method_name: str
    ) -> List[str]:
        if not self._is_allowed_class(qualified_class_name):
            return []
        return super().get_method_parameters(
            qualified_class_name, qualified_method_name
        )

    def get_java_file(self, qualified_class_name: str) -> str | None:
        if not self._is_allowed_class(qualified_class_name):
            return None
        return super().get_java_file(qualified_class_name)

    def get_methods_in_class(self, qualified_class_name: str) -> Dict[str, JCallable]:
        if not self._is_allowed_class(qualified_class_name):
            return {}
        return super().get_methods_in_class(qualified_class_name)

    def get_constructors(self, qualified_class_name: str) -> Dict[str, JCallable]:
        if not self._is_allowed_class(qualified_class_name):
            return {}
        return super().get_constructors(qualified_class_name)

    def get_fields(self, qualified_class_name: str) -> List[JField]:
        if not self._is_allowed_class(qualified_class_name):
            return []
        return super().get_fields(qualified_class_name)

    def get_nested_classes(self, qualified_class_name: str) -> List[JType]:
        if not self._is_allowed_class(qualified_class_name):
            return []
        return super().get_nested_classes(qualified_class_name)

    def get_sub_classes(self, qualified_class_name: str) -> Dict[str, JType]:
        if not self._is_allowed_class(qualified_class_name):
            return {}
        return super().get_sub_classes(qualified_class_name)

    def get_extended_classes(self, qualified_class_name: str) -> List[str]:
        if not self._is_allowed_class(qualified_class_name):
            return []
        return super().get_extended_classes(qualified_class_name)

    def get_implemented_interfaces(self, qualified_class_name: str) -> List[str]:
        if not self._is_allowed_class(qualified_class_name):
            return []
        return super().get_implemented_interfaces(qualified_class_name)

    def get_callers(
        self,
        target_class_name: str,
        target_method_declaration: str,
        using_symbol_table: bool = False,
    ) -> dict[str, object]:
        if not self._is_allowed_class(target_class_name):
            return {}
        return super().get_callers(
            target_class_name,
            target_method_declaration,
            using_symbol_table,
        )

    def get_callees(
        self,
        source_class_name: str,
        source_method_declaration: str,
        using_symbol_table: bool = False,
    ) -> dict[str, object]:
        if not self._is_allowed_class(source_class_name):
            return {}
        return super().get_callees(
            source_class_name,
            source_method_declaration,
            using_symbol_table,
        )

    def get_class_call_graph(
        self,
        qualified_class_name: str,
        method_signature: str | None = None,
        using_symbol_table: bool = False,
    ) -> List[Tuple[JMethodDetail, JMethodDetail]]:
        if not self._is_allowed_class(qualified_class_name):
            return []
        return super().get_class_call_graph(
            qualified_class_name,
            method_signature=method_signature,
            using_symbol_table=using_symbol_table,
        )

    def get_comments_in_a_method(
        self, qualified_class_name: str, method_signature: str
    ) -> List[JComment]:
        if not self._is_allowed_class(qualified_class_name):
            return []
        return super().get_comments_in_a_method(
            qualified_class_name, method_signature
        )

    def get_comments_in_a_class(self, qualified_class_name: str) -> List[JComment]:
        if not self._is_allowed_class(qualified_class_name):
            return []
        return super().get_comments_in_a_class(qualified_class_name)
