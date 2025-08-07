from typing import List

from cldk.analysis.java import JavaAnalysis

from nltest.test2nl.model.models import GeneratedDescription, RoundTripTest
from nltest.test2nl.prompt import RoundTripPrompt
from nltest.utils import Config


class RoundTripGenerator:
    def __init__(self, analysis: JavaAnalysis) -> None:

        # Get temperature from config
        config = Config()
        llm_provider = config.get("llm_provider", "name")
        self.temp = config.get(llm_provider, "code_gen_temp")

        self.roundtrip_prompt = RoundTripPrompt(analysis)

    def generate(self, gen_descriptions: List[GeneratedDescription]) -> List[RoundTripTest]:
        roundtrip_tests: List[RoundTripTest] = []
        for gen_desc in gen_descriptions:
            gen_test, prompt, is_successful = self.roundtrip_prompt.generate(
                gen_desc.method_signature,
                gen_desc.qualified_class_name,
                gen_desc.description
            )

            if is_successful:
                roundtrip_tests.append(
                    RoundTripTest(
                        prompt=prompt,
                        temperature=self.temp,
                        generated_test=gen_test,
                        method_signature=gen_desc.method_signature,
                        qualified_class_name=gen_desc.qualified_class_name,
                        generated_description=gen_desc,
                    )
                )

        return roundtrip_tests
