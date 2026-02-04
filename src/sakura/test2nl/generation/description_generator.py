from typing import Dict, List

from cldk.analysis.java import JavaAnalysis
from hamster.code_analysis.model.models import TestingFramework

from sakura.test2nl.model.models import AbstractionLevel, TestDescriptionInfo
from sakura.test2nl.prompts import Test2NLPrompt
from sakura.utils.analysis import CommonAnalysis, Reachability


class DescriptionGenerator:
    def __init__(
        self,
        analysis: JavaAnalysis,
        application_classes: List[str] | None = None,
        test_utility_classes: List[str] | None = None,
    ) -> None:
        self.analysis = analysis
        self._application_classes = application_classes if application_classes else []
        self._test_utility_classes = (
            test_utility_classes if test_utility_classes else []
        )
        self.test2nl_prompt = Test2NLPrompt(
            self.analysis, self._application_classes, self._test_utility_classes
        )

    def generate_for_method(
        self,
        method_signature: str,
        qualified_class_name: str,
        abstraction: AbstractionLevel = AbstractionLevel.HIGH,
    ) -> TestDescriptionInfo | None:
        temperature = abstraction.get_temperature()
        test_desc, prompt, is_successful = self.test2nl_prompt.generate(
            method_signature, qualified_class_name, abstraction, temperature=temperature
        )

        if not is_successful or test_desc is None:
            return None

        generated_description = TestDescriptionInfo(
            description=test_desc,
            prompt=prompt,
            abstraction_level=abstraction,
            temperature=temperature,
            method_signature=method_signature,
            qualified_class_name=qualified_class_name,
        )

        return generated_description

    def generate_for_class(
        self,
        qualified_class_name: str,
        testing_frameworks: List[TestingFramework],
        abstraction: AbstractionLevel = AbstractionLevel.HIGH,
        max_entries: int = 0,
    ) -> List[TestDescriptionInfo]:
        generated_descriptions: List[TestDescriptionInfo] = []

        # Get all reachable test methods (direct + inherited)
        reachable_test_methods: Dict[str, List[str]] = Reachability(
            self.analysis
        ).get_reachable_test_methods(qualified_class_name, testing_frameworks)

        for declaring_class, method_signatures in reachable_test_methods.items():
            for method_signature in method_signatures:
                if max_entries > 0 and len(generated_descriptions) >= max_entries:
                    return generated_descriptions

                # Retrieve method from declaring class
                generated_description = self.generate_for_method(
                    method_signature, declaring_class, abstraction
                )
                if generated_description:
                    # Report under concrete class name (containing class)
                    generated_description.qualified_class_name = qualified_class_name
                    generated_descriptions.append(generated_description)

        return generated_descriptions

    def generate_for_project(
        self,
        abstraction: AbstractionLevel = AbstractionLevel.HIGH,
        max_entries: int = 0,
        only_interesting_tests: bool = False,
    ) -> List[TestDescriptionInfo]:
        generated_descriptions: List[TestDescriptionInfo] = []
        common_analysis = CommonAnalysis(self.analysis)

        if only_interesting_tests:
            complicated_tests = common_analysis.get_complicated_focal_tests()

            for (
                qualified_class_name,
                interesting_method_signatures,
            ) in complicated_tests.items():
                if max_entries > 0 and len(generated_descriptions) >= max_entries:
                    break

                # Skip abstract classes
                if common_analysis.is_abstract_class(qualified_class_name):
                    continue

                testing_frameworks = common_analysis.get_testing_frameworks_for_class(
                    qualified_class_name
                )

                for method_signature in interesting_method_signatures:
                    if max_entries > 0 and len(generated_descriptions) >= max_entries:
                        break

                    generated_description = self.generate_for_method(
                        method_signature, qualified_class_name, abstraction
                    )
                    if generated_description:
                        generated_descriptions.append(generated_description)
        else:
            for qualified_class_name in self.analysis.get_classes():
                if max_entries > 0 and len(generated_descriptions) >= max_entries:
                    break

                # Skip abstract classes
                if common_analysis.is_abstract_class(qualified_class_name):
                    continue

                testing_frameworks = common_analysis.get_testing_frameworks_for_class(
                    qualified_class_name
                )
                if common_analysis.is_test_class(
                    qualified_class_name, testing_frameworks
                ):
                    remaining_entries = (
                        max_entries - len(generated_descriptions)
                        if max_entries > 0
                        else 0
                    )
                    class_descriptions = self.generate_for_class(
                        qualified_class_name,
                        testing_frameworks,
                        abstraction,
                        remaining_entries,
                    )
                    generated_descriptions.extend(class_descriptions)

        return generated_descriptions

    def generate(
        self,
        abstraction_level: AbstractionLevel = AbstractionLevel.HIGH,
        max_entries: int = 0,
        only_interesting_tests: bool = False,
    ) -> List[TestDescriptionInfo]:
        return self.generate_for_project(
            abstraction_level, max_entries, only_interesting_tests
        )
