from typing import List

from cldk.analysis.java import JavaAnalysis

from nltest.test2nl.model.models import TestDescriptionInfo, RoundTripTest
from nltest.test2nl.prompt import RoundTripPrompt
from nltest.utils import Config


class RoundTripGenerator:
    def __init__(self, analysis: JavaAnalysis) -> None:

        # Get temperature from config
        config = Config()
        llm_provider = config.get("llm_provider", "name")
        self.temp = config.get(llm_provider, "code_gen_temp")

        self.roundtrip_prompt = RoundTripPrompt(analysis)

    def generate(self, test_descriptions: List[TestDescriptionInfo]) -> List[RoundTripTest]:
        roundtrip_tests: List[RoundTripTest] = []
        for test_description_info in test_descriptions:
            gen_test, prompt, is_successful = self.roundtrip_prompt.generate(
                test_description_info.method_signature,
                test_description_info.qualified_class_name,
                test_description_info.description
            )

            if is_successful:
                roundtrip_tests.append(
                    RoundTripTest(
                        prompt=prompt,
                        temperature=self.temp,
                        generated_test=gen_test,
                        method_signature=test_description_info.method_signature,
                        qualified_class_name=test_description_info.qualified_class_name,
                        generated_description=test_description_info,
                    )
                )

        return roundtrip_tests
