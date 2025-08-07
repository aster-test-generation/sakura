from typing import List

from cldk.analysis.java import JavaAnalysis
from hamster.code_analysis.model.models import TestingFramework

from nltest.test2nl.model.models import AbstractionLevel, GeneratedDescription
from nltest.test2nl.prompt import Test2NLPrompt
from nltest.utils import Config
from nltest.utils.analysis import CommonAnalysis


class DescriptionGenerator:
    def __init__(self, analysis: JavaAnalysis) -> None:
        # Get model temperature from config
        config = Config()
        llm_provider = config.get("llm_provider", "name")
        self.temp = config.get(llm_provider, "code_gen_temp")

        self.analysis = analysis

        self.test2nl_prompt = Test2NLPrompt(self.analysis)

    def generate_for_method(self, method_signature: str, qualified_class_name: str,
                            abstraction: AbstractionLevel = AbstractionLevel.HIGH) -> GeneratedDescription | None:
        test_desc, prompt, is_successful = self.test2nl_prompt.generate(method_signature, qualified_class_name,
                                                                        abstraction)

        if not is_successful:
            return None

        generated_description = GeneratedDescription(
            description=test_desc,
            prompt=prompt,
            abstraction_level=abstraction,
            temperature=self.temp,
            method_signature=method_signature,
            qualified_class_name=qualified_class_name,
        )

        return generated_description

    def generate_for_class(self, qualified_class_name: str, testing_frameworks: List[TestingFramework],
                           abstraction: AbstractionLevel = AbstractionLevel.HIGH) -> List[GeneratedDescription]:
        generated_descriptions: List[GeneratedDescription] = []

        for method_signature in self.analysis.get_methods_in_class(qualified_class_name):
            if CommonAnalysis(self.analysis).is_test_method(method_signature, qualified_class_name, testing_frameworks):

                generated_description = self.generate_for_method(method_signature, qualified_class_name, abstraction)
                if generated_description:
                    generated_descriptions.append(generated_description)

        return generated_descriptions

    def generate_for_project(self, abstraction: AbstractionLevel = AbstractionLevel.HIGH) -> List[GeneratedDescription]:
        generated_descriptions: List[GeneratedDescription] = []

        for qualified_class_name in self.analysis.get_classes():
            testing_frameworks = CommonAnalysis(self.analysis).get_testing_frameworks_for_class(qualified_class_name)
            if CommonAnalysis(self.analysis).is_test_class(qualified_class_name, testing_frameworks):
                generated_descriptions.extend(
                    self.generate_for_class(qualified_class_name, testing_frameworks, abstraction)
                )

        return generated_descriptions

    def generate(self, abstraction_level: AbstractionLevel = AbstractionLevel.HIGH) -> List[GeneratedDescription]:
        return self.generate_for_project(abstraction_level)
