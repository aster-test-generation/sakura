from typing import List

from cldk.analysis.java import JavaAnalysis
from hamster.code_analysis.model.models import TestingFramework

from nltest.test2nl.model.models import AbstractionLevel, TestDescriptionInfo
from nltest.test2nl.prompts import Test2NLPrompt
from nltest.utils.config import Config
from nltest.utils.analysis import CommonAnalysis


class DescriptionGenerator:
    def __init__(self, analysis: JavaAnalysis) -> None:
        # Get model temperature from config
        config = Config()
        self.temp = config.get("llm", "code_gen_temp")

        self.analysis = analysis

        self.test2nl_prompt = Test2NLPrompt(self.analysis)

    def generate_for_method(self, method_signature: str, qualified_class_name: str,
                            abstraction: AbstractionLevel = AbstractionLevel.HIGH) -> TestDescriptionInfo | None:
        test_desc, prompt, is_successful = self.test2nl_prompt.generate(method_signature, qualified_class_name,
                                                                        abstraction)

        if not is_successful:
            return None

        generated_description = TestDescriptionInfo(
            description=test_desc,
            prompt=prompt,
            abstraction_level=abstraction,
            temperature=self.temp,
            method_signature=method_signature,
            qualified_class_name=qualified_class_name,
        )

        return generated_description

    def generate_for_class(self, qualified_class_name: str, testing_frameworks: List[TestingFramework],
                           abstraction: AbstractionLevel = AbstractionLevel.HIGH, max_entries: int = 0) -> List[TestDescriptionInfo]:
        generated_descriptions: List[TestDescriptionInfo] = []

        for method_signature in self.analysis.get_methods_in_class(qualified_class_name):
            # Check if we've reached the limit
            if max_entries > 0 and len(generated_descriptions) >= max_entries:
                break
                
            if CommonAnalysis(self.analysis).is_test_method(method_signature, qualified_class_name, testing_frameworks):
                generated_description = self.generate_for_method(method_signature, qualified_class_name, abstraction)
                if generated_description:
                    generated_descriptions.append(generated_description)

        return generated_descriptions

    def generate_for_project(self, abstraction: AbstractionLevel = AbstractionLevel.HIGH, max_entries: int = 0, only_interesting_tests: bool = False) -> List[TestDescriptionInfo]:
        generated_descriptions: List[TestDescriptionInfo] = []

        if only_interesting_tests:
            # Use complicated focal tests instead of all tests
            complicated_tests = CommonAnalysis(self.analysis).get_complicated_focal_tests()
            
            for qualified_class_name, interesting_method_signatures in complicated_tests.items():
                if max_entries > 0 and len(generated_descriptions) >= max_entries:
                    break
                    
                testing_frameworks = CommonAnalysis(self.analysis).get_testing_frameworks_for_class(qualified_class_name)
                
                for method_signature in interesting_method_signatures:
                    # Check if we've reached the limit
                    if max_entries > 0 and len(generated_descriptions) >= max_entries:
                        break
                        
                    generated_description = self.generate_for_method(method_signature, qualified_class_name, abstraction)
                    if generated_description:
                        generated_descriptions.append(generated_description)
        else:
            # Original logic for all tests
            for qualified_class_name in self.analysis.get_classes():
                # Check if we've reached the limit
                if max_entries > 0 and len(generated_descriptions) >= max_entries:
                    break
                    
                testing_frameworks = CommonAnalysis(self.analysis).get_testing_frameworks_for_class(qualified_class_name)
                if CommonAnalysis(self.analysis).is_test_class(qualified_class_name, testing_frameworks):
                    # Calculate remaining entries we can generate
                    remaining_entries = max_entries - len(generated_descriptions) if max_entries > 0 else 0
                    class_descriptions = self.generate_for_class(qualified_class_name, testing_frameworks, abstraction, remaining_entries)
                    generated_descriptions.extend(class_descriptions)

        return generated_descriptions

    def generate(self, abstraction_level: AbstractionLevel = AbstractionLevel.HIGH, max_entries: int = 0, only_interesting_tests: bool = False) -> List[TestDescriptionInfo]:
        return self.generate_for_project(abstraction_level, max_entries, only_interesting_tests)
