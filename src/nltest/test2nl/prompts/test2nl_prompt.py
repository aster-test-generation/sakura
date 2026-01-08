import json
from typing import List, Tuple

from cldk.analysis.java import JavaAnalysis
from cldk.models.java import JCallable

from nltest.test2nl.extractors import (
    ClassExtractor,
    FieldDeclarationExtractor,
    MethodExtractor,
)
from nltest.test2nl.model.models import (
    AbstractionLevel,
    ClassContext,
    FieldDeclaration,
    MethodContext,
)
from nltest.test2nl.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.utils.analysis import CommonAnalysis, Reachability
from nltest.utils.llm import ClientType, LLMClient


class Test2NLPrompt:
    def __init__(
        self,
        analysis: JavaAnalysis,
        application_classes: List[str] | None = None,
        test_utility_classes: List[str] | None = None,
    ):
        super().__init__()
        self.analysis = analysis
        self.llm = LLMClient(ClientType.SUMMARIZATION)
        self.application_classes = application_classes if application_classes else []
        self.test_utility_classes = test_utility_classes if test_utility_classes else []
        self.method_extractor = MethodExtractor(self.analysis, self.application_classes)
        self.common_analysis = CommonAnalysis(self.analysis)
        self.reachability = Reachability(self.analysis)
        self.class_extractor = ClassExtractor(self.analysis, self.application_classes)
        self.field_extractor = FieldDeclarationExtractor(
            self.analysis, self.application_classes
        )

    def format(
        self,
        method_signature: str,
        qualified_class_name: str,
        abstraction_level: AbstractionLevel,
    ) -> str:
        abs_level = abstraction_level.value

        # Get the test method being described
        method_details = self.analysis.get_method(
            qualified_class_name, method_signature
        )
        if not method_details:
            raise Exception(
                f"Method {method_signature} in {qualified_class_name} not found"
            )

        # 1) Get test class context

        class_details = self.analysis.get_class(qualified_class_name)

        # 1.a) Collect test class field declarations
        field_declarations: List[FieldDeclaration] = []
        for field_declaration in class_details.field_declarations:
            field_declarations.append(self.field_extractor.extract(field_declaration))

        # 1.b) Collect test class annotations
        class_annotations: List[str] = (
            class_details.annotations if class_details.annotations else []
        )

        # 2) Collect setup methods (@Before, @BeforeEach, etc.)
        setup_methods: List[MethodContext] = []
        setup_methods_dict = self.common_analysis.get_setup_methods(
            qualified_class_name
        )
        for declaring_class, method_sigs in setup_methods_dict.items():
            is_inherited = declaring_class != qualified_class_name
            for method_sig in method_sigs:
                setup_method_details = self.analysis.get_method(
                    declaring_class, method_sig
                )
                if setup_method_details:
                    setup_methods.append(
                        self.method_extractor.extract(
                            declaring_class,
                            method_sig,
                            complete_methods=True,
                            include_class_name=is_inherited,
                        )
                    )

        # 3) Collect teardown methods (@After, @AfterEach, etc.)
        teardown_methods: List[MethodContext] = []
        teardown_methods_dict = self.common_analysis.get_teardown_methods(
            qualified_class_name
        )
        for declaring_class, method_sigs in teardown_methods_dict.items():
            is_inherited = declaring_class != qualified_class_name
            for method_sig in method_sigs:
                teardown_method_details = self.analysis.get_method(
                    declaring_class, method_sig
                )
                if teardown_method_details:
                    teardown_methods.append(
                        self.method_extractor.extract(
                            declaring_class,
                            method_sig,
                            complete_methods=True,
                            include_class_name=is_inherited,
                        )
                    )

        # 4) Get test method context

        # 4.a) Collect test method annotations
        method_annotations: List[str] = (
            method_details.annotations if method_details.annotations else []
        )

        # 4.b) Collect test method details
        method: MethodContext = self.method_extractor.extract(
            qualified_class_name, method_signature, complete_methods=True
        )

        # 5) Collect helper context across setup, teardown, and test

        # 5.a) Collect helper methods using call sites
        helper_methods: List[MethodContext] = []
        seen_helpers: set[tuple[str, str]] = set()

        def collect_helpers_from_method(
            source_class: str, source_method_sig: str
        ) -> None:
            """Collect helper methods from a given method."""
            if not self.analysis.get_method(source_class, source_method_sig):
                return
            helpers_dict = self.reachability.get_helper_methods(
                source_class,
                source_method_sig,
                depth=1,
                add_extended_class=True,
                test_utility_classes=self.test_utility_classes,
            )
            for helper_class, helper_sigs in helpers_dict.items():
                for helper_sig in helper_sigs:
                    key = (helper_class, helper_sig)
                    if key in seen_helpers:
                        continue
                    seen_helpers.add(key)
                    helper_details = self.analysis.get_method(helper_class, helper_sig)
                    if not helper_details:
                        continue
                    helper_context = self.method_extractor.extract(
                        helper_class,
                        helper_sig,
                        complete_methods=True,
                        include_class_name=True,
                    )
                    helper_methods.append(helper_context)

        collect_helpers_from_method(qualified_class_name, method_signature)
        for setup_method in setup_methods:
            source_class = setup_method.qualified_class_name or qualified_class_name
            collect_helpers_from_method(source_class, setup_method.method_signature)
        for teardown_method in teardown_methods:
            source_class = teardown_method.qualified_class_name or qualified_class_name
            collect_helpers_from_method(source_class, teardown_method.method_signature)

        # 5.b) Collect test utility classes using referenced classes
        # NOTE: Not implemented - unlikely scenario

        # 6) Get application context

        def collect_app_context_from_callable(
            callable_details: JCallable,
        ) -> dict[str, set[str]]:
            """
            Collect application class references from a JCallable's call sites
            and referenced types.
            Returns dict mapping qualified class name -> set of called method names.
            """
            app_class_methods: dict[str, set[str]] = {}
            for cs in callable_details.call_sites or []:
                if cs.receiver_type in self.application_classes:
                    app_class_methods.setdefault(cs.receiver_type, set()).add(
                        cs.method_name
                    )
            for ref_type in callable_details.referenced_types or []:
                non_param_types = CommonAnalysis.extract_non_parameterized_types(
                    ref_type
                )
                for t in non_param_types:
                    if t in self.application_classes:
                        app_class_methods.setdefault(t, set())
            return app_class_methods

        def get_callable_from_method_context(
            method_ctx: MethodContext, default_class: str
        ) -> JCallable | None:
            """Get JCallable from MethodContext using its class name or default."""
            cls = method_ctx.qualified_class_name or default_class
            return self.analysis.get_method(cls, method_ctx.method_signature)

        def merge_app_context(
            target: dict[str, set[str]], source: dict[str, set[str]]
        ) -> None:
            """Merge source app context into target."""
            for cls, methods in source.items():
                target.setdefault(cls, set()).update(methods)

        app_class_context: dict[str, set[str]] = {}

        # 6.a) Get application context from helpers
        for helper in helper_methods:
            helper_callable = get_callable_from_method_context(
                helper, qualified_class_name
            )
            if helper_callable:
                merge_app_context(
                    app_class_context,
                    collect_app_context_from_callable(helper_callable),
                )

        # 6.b) Get application context from test method
        merge_app_context(
            app_class_context, collect_app_context_from_callable(method_details)
        )

        # 6.c) Get application context from setup and teardown
        for setup_method in setup_methods:
            setup_callable = get_callable_from_method_context(
                setup_method, qualified_class_name
            )
            if setup_callable:
                merge_app_context(
                    app_class_context, collect_app_context_from_callable(setup_callable)
                )
        for teardown_method in teardown_methods:
            teardown_callable = get_callable_from_method_context(
                teardown_method, qualified_class_name
            )
            if teardown_callable:
                merge_app_context(
                    app_class_context,
                    collect_app_context_from_callable(teardown_callable),
                )

        # 6.d) Extract application classes using collected context
        application_classes: List[ClassContext] = []
        for app_qualified_class_name, called_methods in app_class_context.items():
            if not self.analysis.get_class(app_qualified_class_name):
                continue
            application_classes.append(
                self.class_extractor.extract(
                    app_qualified_class_name,
                    complete_methods=False,  # Application code should not be included in its entirety
                    called_method_names=called_methods,  # Passes empty set for referenced types, so that only constructors are kept
                )
            )

        # Serialize all context to JSON for prompt template
        method_context_str: str = json.dumps(
            method.model_dump(exclude_none=True), separators=(",", ":")
        )
        setup_methods_str: List[str] = [
            json.dumps(
                setup_context.model_dump(exclude_none=True), separators=(",", ":")
            )
            for setup_context in setup_methods
        ]
        teardown_methods_str: List[str] = [
            json.dumps(
                teardown_context.model_dump(exclude_none=True), separators=(",", ":")
            )
            for teardown_context in teardown_methods
        ]
        field_declarations_str: List[str] = [
            json.dumps(
                field_context.model_dump(exclude_none=True), separators=(",", ":")
            )
            for field_context in field_declarations
        ]
        helper_methods_str: List[str] = [
            json.dumps(
                helper_context.model_dump(exclude_none=True), separators=(",", ":")
            )
            for helper_context in helper_methods
        ]
        referenced_classes_str: List[str] = [
            json.dumps(
                referenced_context.model_dump(exclude_none=True), separators=(",", ":")
            )
            for referenced_context in application_classes
        ]
        class_annotation_str: str = (
            ", ".join(class_annotations) if class_annotations else "None"
        )
        method_annotation_str: str = (
            ", ".join(method_annotations) if method_annotations else "None"
        )

        # Render the prompt template with all collected context
        chat_template = LoadPrompt.load_jinja2_template(
            f"{abs_level}_abs.jinja2", "chat"
        )
        rendered_prompt = chat_template.render(
            class_annotations=class_annotation_str,
            field_declarations=field_declarations_str,
            setup_methods=setup_methods_str,
            method_annotations=method_annotation_str,
            test_method=method_context_str,
            helper_methods=helper_methods_str,
            teardown_methods=teardown_methods_str,
            application_classes=referenced_classes_str,
        )
        return rendered_prompt

    def generate(
        self,
        method_signature: str,
        qualified_class_name: str,
        abstraction_level: AbstractionLevel,
        temperature: float | None = None,
    ) -> Tuple[str | None, str, bool]:
        """
        Prompt to get the natural language description of a test case.
        Args:
            method_signature: The method signature of the test case
            qualified_class_name: The qualified class name containing the method with the test case.
            abstraction_level: The abstraction level of the natural language description.
            temperature: Optional temperature override for the LLM call.

        Returns:
            Tuple[str|None, str, bool]: The natural language description of the test case, the prompts, and
            a boolean indicating if the test case was successful.

        """

        system_prompt = LoadPrompt.load_prompt(
            f"{abstraction_level.value}_abs.jinja2", PromptFormat.JINJA2, "system"
        ).format()
        chat_prompt = self.format(
            method_signature, qualified_class_name, abstraction_level
        )

        ai_msg = self.llm.invoke_prompts(
            system_prompt, chat_prompt, temperature=temperature
        )
        test_desc = ai_msg.content.strip() if ai_msg and ai_msg.content else None

        if test_desc:
            return test_desc, chat_prompt, True
        return None, chat_prompt, False
