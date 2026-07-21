import json
from typing import List, Tuple

from cldk.analysis.java import JavaAnalysis

from sakura.test2nl.context import Test2NLContextBuilder
from sakura.test2nl.model.models import AbstractionLevel, Test2NLContext
from sakura.test2nl.prompts.load_prompt import LoadPrompt, PromptFormat
from sakura.utils.llm import ClientType, LLMClient


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
        self.context_builder = Test2NLContextBuilder(
            analysis, application_classes, test_utility_classes
        )

    @staticmethod
    def _json_context(items: list) -> list[str]:
        return [
            json.dumps(item.model_dump(exclude_none=True), separators=(",", ":"))
            for item in items
        ]

    def format(
        self,
        method_signature: str,
        qualified_class_name: str,
        abstraction_level: AbstractionLevel,
    ) -> str:
        context: Test2NLContext = self.context_builder.build(
            method_signature, qualified_class_name
        )
        method_context = json.dumps(
            context.test_method.model_dump(exclude_none=True), separators=(",", ":")
        )
        class_annotations = (
            ", ".join(context.class_annotations)
            if context.class_annotations
            else "None"
        )
        method_annotations = (
            ", ".join(context.method_annotations)
            if context.method_annotations
            else "None"
        )

        chat_template = LoadPrompt.load_jinja2_template(
            f"{abstraction_level.value}_abs.jinja2", "chat"
        )
        return chat_template.render(
            class_annotations=class_annotations,
            field_declarations=self._json_context(context.field_declarations),
            setup_methods=self._json_context(context.setup_methods),
            method_annotations=method_annotations,
            test_method=method_context,
            helper_methods=self._json_context(context.helper_methods),
            teardown_methods=self._json_context(context.teardown_methods),
            application_classes=self._json_context(context.application_classes),
        )

    def generate(
        self,
        method_signature: str,
        qualified_class_name: str,
        abstraction_level: AbstractionLevel,
        temperature: float | None = None,
    ) -> Tuple[str | None, str, bool]:
        """Generate a natural-language description of one test case."""
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
