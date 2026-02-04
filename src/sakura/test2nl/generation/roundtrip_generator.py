from typing import List

from cldk.analysis.java import JavaAnalysis

from sakura.test2nl.model.models import TestDescriptionInfo, RoundTripTest
from sakura.test2nl.prompts import RoundTripPrompt
from sakura.utils.config import Config


class RoundTripGenerator:
    # DEPRECATED: Roundtrip generation is part of the legacy pipeline.
    def __init__(self, analysis: JavaAnalysis) -> None:
        # Get temperature from config
        config = Config()
        self.temp = config.get("llm", "code_gen_temp")

        self.roundtrip_prompt = RoundTripPrompt(analysis)

    def generate(
        self, test_descriptions: List[TestDescriptionInfo]
    ) -> List[RoundTripTest]:
        roundtrip_tests: List[RoundTripTest] = []
        for test_description_info in test_descriptions:
            gen_test, prompt, is_successful = self.roundtrip_prompt.generate(
                test_description_info.method_signature,
                test_description_info.qualified_class_name,
                test_description_info.description,
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
