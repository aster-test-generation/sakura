from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from nltest.utils.llm import LLMClient
from nltest.utils.llm.model import ClientType
from nltest.utils.pretty.prints import pretty_print

from nltest.test2nl.model.models import AbstractionLevel, TestDescriptionInfo

from tests._base_nl2test import BaseNL2Test
from tests._base_test2nl import BaseTest2NL


class TestLLMClient(BaseNL2Test):
    def test_llm_client(self):
        llm = LLMClient(ClientType.SUMMARIZATION)

        system = "You are a helpful assistant that provides eloquent summaries with a British accent."
        query = "Why do people attend concerts? What is the purpose if they can just listen to the music at home?"

        messages = [SystemMessage(content=system), HumanMessage(content=query)]

        result = llm.invoke_messages(messages)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, AIMessage)
        pretty_print("LLM Output", result)


class TestPrompts(BaseTest2NL):
    def test_high_desc_prompt(self):
        qualified_class_name = (
            "org.springframework.samples.petclinic.service.ClinicServiceTests"
        )
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        test_desc, prompt, is_successful = self.test2nl_prompt.generate(
            method_signature, qualified_class_name, AbstractionLevel.HIGH
        )
        self.assertTrue(prompt, "Prompt was unsuccessfully rendered...")

        pretty_print("prompts", prompt)

        self.assertTrue(is_successful, "LLM generated was unsuccessful with prompts...")

    def test_roundtrip_prompt(self):
        qualified_class_name = (
            "org.springframework.samples.petclinic.service.ClinicServiceTests"
        )
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        test_descriptions = self.data_manager.load(
            "descriptions.json", TestDescriptionInfo
        )
        selected_description = test_descriptions[0]

        test_case, prompt, is_successful = self.roundtrip_prompt.generate(
            method_signature, qualified_class_name, selected_description.description
        )
        self.assertTrue(prompt, "Prompt was unsuccessfully rendered...")

        pretty_print("prompts", prompt)

        self.assertTrue(is_successful, "LLM generated was unsuccessful with prompts...")
