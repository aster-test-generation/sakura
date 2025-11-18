from typing import Tuple, List

import yaml
from cldk.analysis.java import JavaAnalysis

from nltest.test2nl.model.models import ReferencedClass
from nltest.test2nl.extractors import ReferencedClassExtractor
from nltest.test2nl.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.utils.analysis import CommonAnalysis
from nltest.utils.llm import LLMClient, ClientType


class RoundTripPrompt:
    """
    This class is used to generate the roundtrip prompt for the test case.

    TODO: Update prompt.
    """
    def __init__(self, analysis: JavaAnalysis):
        super().__init__()
        self.analysis = analysis
        self.llm = LLMClient(ClientType.CODE_GEN)

    def format(self, test_method_signature: str, test_qualified_class: str, description: str) -> str:
        # Base prompts needs natural language description, test class imports, test class field declarations,
        # and the custom classes

        method_details = self.analysis.get_method(test_qualified_class, test_method_signature)

        # Get all referenced classes from the test method (relevant code context)
        referenced_classes: List[ReferencedClass] = []
        ref_class_names: List[str] = CommonAnalysis(self.analysis).get_referenced_app_classes(method_details)
        for qualified_class_name in ref_class_names:
            if qualified_class_name != test_qualified_class:
                referenced_classes.append(
                    ReferencedClassExtractor(self.analysis).extract(qualified_class_name, complete_methods=True)
                )
        referenced_class_str: List[str] = [yaml.dump(referenced_class.model_dump(), sort_keys=False, indent=4) for
                                           referenced_class in referenced_classes]

        prompt_template = LoadPrompt.load_prompt("roundtrip_prompt.jinja2", PromptFormat.JINJA2)
        rendered_prompt = prompt_template.format(
            test_case_description=description,
            custom_classes=referenced_class_str,
        )
        return rendered_prompt

    def generate(self, test_method_signature: str, test_qualified_class: str,
                 description: str) -> Tuple[str | None, str, bool]:
        rendered_prompt = self.format(test_method_signature, test_qualified_class, description)

        # Call the LLM
        test_case = self.llm.generate(rendered_prompt, sanitize=True)
        # Strip the ```java ```
        # test_block = FormatValidator.strip_java_block(test_case)
        test_block = test_case.strip()
        if test_block:
            return test_block, rendered_prompt, True
        return None, rendered_prompt, False
