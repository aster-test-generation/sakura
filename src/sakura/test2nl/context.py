from __future__ import annotations

from typing import List

from cldk.analysis.java import JavaAnalysis
from cldk.models.java import JCallable

from sakura.test2nl.extractors import (
    ClassExtractor,
    FieldDeclarationExtractor,
    MethodExtractor,
)
from sakura.test2nl.model.models import (
    ClassContext,
    FieldDeclaration,
    MethodContext,
    Test2NLContext,
)
from sakura.utils.analysis import CommonAnalysis, Reachability


class Test2NLContextBuilder:
    """Collect the exact source context used to construct a Test2NL prompt."""

    def __init__(
        self,
        analysis: JavaAnalysis,
        application_classes: List[str] | None = None,
        test_utility_classes: List[str] | None = None,
    ) -> None:
        self.analysis = analysis
        self.application_classes = application_classes or []
        self.test_utility_classes = test_utility_classes or []
        self.method_extractor = MethodExtractor(analysis, self.application_classes)
        self.common_analysis = CommonAnalysis(analysis)
        self.reachability = Reachability(analysis)
        self.class_extractor = ClassExtractor(analysis, self.application_classes)
        self.field_extractor = FieldDeclarationExtractor(
            analysis, self.application_classes
        )

    def build(
        self, method_signature: str, qualified_class_name: str
    ) -> Test2NLContext:
        method_details = self.analysis.get_method(
            qualified_class_name, method_signature
        )
        if not method_details:
            raise ValueError(
                f"Method {method_signature} in {qualified_class_name} not found"
            )

        class_details = self.analysis.get_class(qualified_class_name)
        if not class_details:
            raise ValueError(f"Class {qualified_class_name} not found")

        field_declarations: list[FieldDeclaration] = [
            self.field_extractor.extract(field)
            for field in class_details.field_declarations
        ]
        class_annotations = class_details.annotations or []

        setup_methods = self._collect_fixtures(
            self.common_analysis.get_setup_methods(qualified_class_name),
            qualified_class_name,
        )
        teardown_methods = self._collect_fixtures(
            self.common_analysis.get_teardown_methods(qualified_class_name),
            qualified_class_name,
        )

        method_annotations = method_details.annotations or []
        method = self.method_extractor.extract(
            qualified_class_name, method_signature, complete_methods=True
        )

        helper_methods: list[MethodContext] = []
        seen_helpers: set[tuple[str, str]] = set()

        def collect_helpers(source_class: str, source_signature: str) -> None:
            if not self.analysis.get_method(source_class, source_signature):
                return
            helpers = self.reachability.get_helper_methods(
                source_class,
                source_signature,
                depth=1,
                add_extended_class=True,
                test_utility_classes=self.test_utility_classes,
            )
            for helper_class, helper_signatures in helpers.items():
                for helper_signature in helper_signatures:
                    key = (helper_class, helper_signature)
                    if key in seen_helpers:
                        continue
                    seen_helpers.add(key)
                    if not self.analysis.get_method(helper_class, helper_signature):
                        continue
                    helper_methods.append(
                        self.method_extractor.extract(
                            helper_class,
                            helper_signature,
                            complete_methods=True,
                            include_class_name=True,
                        )
                    )

        collect_helpers(qualified_class_name, method_signature)
        for fixture in [*setup_methods, *teardown_methods]:
            collect_helpers(
                fixture.qualified_class_name or qualified_class_name,
                fixture.method_signature,
            )

        application_context: dict[str, set[str]] = {}
        for method_context in helper_methods:
            owner = method_context.qualified_class_name or qualified_class_name
            callable_details = self.analysis.get_method(
                owner, method_context.method_signature
            )
            if callable_details:
                self._merge_application_context(
                    application_context,
                    self._application_context_from_callable(callable_details),
                )
        self._merge_application_context(
            application_context,
            self._application_context_from_callable(method_details),
        )
        for method_context in [*setup_methods, *teardown_methods]:
            owner = method_context.qualified_class_name or qualified_class_name
            callable_details = self.analysis.get_method(
                owner, method_context.method_signature
            )
            if callable_details:
                self._merge_application_context(
                    application_context,
                    self._application_context_from_callable(callable_details),
                )

        application_classes: list[ClassContext] = []
        for application_class, called_methods in application_context.items():
            if not self.analysis.get_class(application_class):
                continue
            application_classes.append(
                self.class_extractor.extract(
                    application_class,
                    complete_methods=False,
                    called_method_names=called_methods,
                )
            )

        return Test2NLContext(
            qualified_class_name=qualified_class_name,
            class_annotations=class_annotations,
            field_declarations=field_declarations,
            setup_methods=setup_methods,
            method_annotations=method_annotations,
            test_method=method,
            helper_methods=helper_methods,
            teardown_methods=teardown_methods,
            application_classes=application_classes,
        )

    def _collect_fixtures(
        self,
        fixtures: dict[str, list[str]],
        qualified_class_name: str,
    ) -> list[MethodContext]:
        result: list[MethodContext] = []
        for declaring_class, method_signatures in fixtures.items():
            is_inherited = declaring_class != qualified_class_name
            for method_signature in method_signatures:
                if not self.analysis.get_method(declaring_class, method_signature):
                    continue
                result.append(
                    self.method_extractor.extract(
                        declaring_class,
                        method_signature,
                        complete_methods=True,
                        include_class_name=is_inherited,
                    )
                )
        return result

    def _application_context_from_callable(
        self, callable_details: JCallable
    ) -> dict[str, set[str]]:
        context: dict[str, set[str]] = {}
        for call_site in callable_details.call_sites or []:
            if call_site.receiver_type in self.application_classes:
                context.setdefault(call_site.receiver_type, set()).add(
                    call_site.method_name
                )
        for referenced_type in callable_details.referenced_types or []:
            for type_name in CommonAnalysis.extract_non_parameterized_types(
                referenced_type
            ):
                if type_name in self.application_classes:
                    context.setdefault(type_name, set())
        return context

    @staticmethod
    def _merge_application_context(
        target: dict[str, set[str]], source: dict[str, set[str]]
    ) -> None:
        for class_name, method_names in source.items():
            target.setdefault(class_name, set()).update(method_names)
