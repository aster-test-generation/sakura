import json
from typing import List, Tuple

from cldk.analysis.java import JavaAnalysis

from nltest.test2nl.extractors import (
    FieldDeclarationExtractor,
    MethodExtractor,
    ReferencedClassExtractor,
)
from nltest.test2nl.model.models import (
    AbstractionLevel,
    FieldDeclaration,
    MethodContext,
    ReferencedClass,
)
from nltest.test2nl.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.utils.analysis import CommonAnalysis, Reachability
from nltest.utils.llm import ClientType, LLMClient


class Test2NLPrompt:
    def __init__(self, analysis: JavaAnalysis):
        super().__init__()
        self.analysis = analysis
        self.llm = LLMClient(ClientType.SUMMARIZATION)

    def format(
        self,
        method_signature: str,
        qualified_class_name: str,
        abstraction_level: AbstractionLevel,
    ) -> str:
        abs_level = abstraction_level.value

        method_details = self.analysis.get_method(
            qualified_class_name, method_signature
        )
        if not method_details:
            raise Exception(
                f"Method {method_signature} in {qualified_class_name} not found"
            )

        setup_methods: List[MethodContext] = []
        setup_methods_dict = CommonAnalysis(self.analysis).get_setup_methods(
            qualified_class_name
        )
        method_extractor = MethodExtractor(self.analysis)
        for declaring_class, method_sigs in setup_methods_dict.items():
            for method_sig in method_sigs:
                setup_method = self.analysis.get_method(declaring_class, method_sig)
                if setup_method:
                    setup_methods.append(
                        method_extractor.extract(
                            declaring_class, method_sig, complete_methods=True
                        )
                    )

        teardown_methods: List[MethodContext] = []
        teardown_methods_dict = CommonAnalysis(self.analysis).get_teardown_methods(
            qualified_class_name
        )
        for declaring_class, method_sigs in teardown_methods_dict.items():
            for method_sig in method_sigs:
                teardown_method = self.analysis.get_method(declaring_class, method_sig)
                if teardown_method:
                    teardown_methods.append(
                        method_extractor.extract(
                            declaring_class, method_sig, complete_methods=True
                        )
                    )

        helper_methods: List[MethodContext] = []
        # Add helper methods from the main test method
        for qualified_class, helper_method_sigs in (
            Reachability(self.analysis)
            .get_helper_methods(qualified_class_name, method_signature, depth=1)
            .items()
        ):
            for helper_sig in helper_method_sigs:
                helper_details = self.analysis.get_method(qualified_class, helper_sig)
                if not helper_details:
                    continue
                helper_context = method_extractor.extract(
                    qualified_class, helper_sig, complete_methods=True
                )
                if helper_context not in helper_methods:
                    helper_methods.append(helper_context)

        # Add helper methods from setup methods
        for setup_method in setup_methods:
            for qualified_class, helper_method_sigs in (
                Reachability(self.analysis)
                .get_helper_methods(
                    qualified_class_name, setup_method.method_signature, depth=1
                )
                .items()
            ):
                for helper_sig in helper_method_sigs:
                    helper_details = self.analysis.get_method(
                        qualified_class, helper_sig
                    )
                    if not helper_details:
                        continue
                    helper_context = method_extractor.extract(
                        qualified_class, helper_sig, complete_methods=True
                    )
                    # Avoid duplicates
                    if helper_context not in helper_methods:
                        helper_methods.append(helper_context)

        # Add helper methods from teardown methods
        for teardown_method in teardown_methods:
            for qualified_class, helper_method_sigs in (
                Reachability(self.analysis)
                .get_helper_methods(
                    qualified_class_name, teardown_method.method_signature, depth=1
                )
                .items()
            ):
                for helper_sig in helper_method_sigs:
                    helper_details = self.analysis.get_method(
                        qualified_class, helper_sig
                    )
                    if not helper_details:
                        continue
                    helper_context = method_extractor.extract(
                        qualified_class, helper_sig, complete_methods=True
                    )
                    # Avoid duplicates
                    if helper_context not in helper_methods:
                        helper_methods.append(helper_context)

        referenced_classes: List[ReferencedClass] = []
        ref_class_names: List[str] = CommonAnalysis(
            self.analysis
        ).get_referenced_app_classes(method_details)
        for ref_qualified_class_name in ref_class_names:
            referenced_classes.append(
                ReferencedClassExtractor(self.analysis).extract(
                    ref_qualified_class_name, complete_methods=True
                )
            )

        method: MethodContext = method_extractor.extract(
            qualified_class_name, method_signature, complete_methods=True
        )

        class_details = self.analysis.get_class(qualified_class_name)

        field_declarations: List[FieldDeclaration] = []
        for field_declaration in class_details.field_declarations:
            field_declarations.append(
                FieldDeclarationExtractor.extract(field_declaration)
            )

        # method_code_str: str = yaml.dump(method.model_dump(), sort_keys=False, indent=4)
        method_code_str: str = json.dumps(method.model_dump(), indent=4)
        # setup_methods_str: List[str] = [yaml.dump(setup_context.model_dump(), sort_keys=False, indent=4) for
        #                                 setup_context in setup_methods]
        setup_methods_str: List[str] = [
            json.dumps(setup_context.model_dump(), indent=4)
            for setup_context in setup_methods
        ]
        # teardown_methods_str: List[str] = [yaml.dump(teardown_context.model_dump(), sort_keys=False, indent=4) for
        #                                    teardown_context in teardown_methods]
        teardown_methods_str: List[str] = [
            json.dumps(teardown_context.model_dump(), indent=4)
            for teardown_context in teardown_methods
        ]
        # field_declarations_str: List[str] = [yaml.dump(field_context.model_dump(), sort_keys=False, indent=4) for
        #                                      field_context in field_declarations]
        field_declarations_str: List[str] = [
            json.dumps(field_context.model_dump(), indent=4)
            for field_context in field_declarations
        ]
        # helper_methods_str: List[str] = [yaml.dump(helper_context.model_dump(), sort_keys=False, indent=4) for
        #                                  helper_context in helper_methods]
        helper_methods_str: List[str] = [
            json.dumps(helper_context.model_dump(), indent=4)
            for helper_context in helper_methods
        ]
        # referenced_classes_str: List[str] = [yaml.dump(referenced_context.model_dump(), sort_keys=False, indent=4) for
        #                                      referenced_context in referenced_classes]
        referenced_classes_str: List[str] = [
            json.dumps(referenced_context.model_dump(), indent=4)
            for referenced_context in referenced_classes
        ]
        class_annotation_str: str = (
            ", ".join(class_details.annotations)
            if class_details.annotations
            else "None"
        )
        method_annotation_str: str = (
            ", ".join(method_details.annotations)
            if method_details.annotations
            else "None"
        )

        chat_template = LoadPrompt.load_jinja2_template(
            f"{abs_level}_abs.jinja2", "chat"
        )
        rendered_prompt = chat_template.render(
            method_code=method_code_str,
            setup_methods=setup_methods_str,
            teardown_methods=teardown_methods_str,
            method_annotations=method_annotation_str,
            class_annotations=class_annotation_str,
            field_declarations=field_declarations_str,
            helper_methods=helper_methods_str,
            custom_classes=referenced_classes_str,
        )
        return rendered_prompt

    def generate(
        self,
        method_signature: str,
        qualified_class_name: str,
        abstraction_level: AbstractionLevel,
    ) -> Tuple[str | None, str, bool]:
        """
        Prompt to get the natural language description of a test case.
        Args:
            method_signature: The method signature of the test case
            qualified_class_name: The qualified class name containing the method with the test case.
            abstraction_level: The abstraction level of the natural language description.

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

        # Call the LLM
        ai_msg = self.llm.invoke_prompts(system_prompt, chat_prompt)
        test_desc = ai_msg.content.strip() if ai_msg and ai_msg.content else None

        if test_desc:
            return test_desc, chat_prompt, True
        return None, chat_prompt, False
