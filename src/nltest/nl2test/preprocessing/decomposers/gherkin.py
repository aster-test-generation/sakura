from __future__ import annotations

from nltest.nl2test.models.decomposition import Scenario
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.nl2test.preprocessing.decomposers.base import BaseDecomposer
from nltest.utils.llm.llm_client import LLMClient, ClientType


class GherkinDecomposer(BaseDecomposer):
    def __init__(self) -> None:
        self.structured = LLMClient(ClientType.STRUCTURED)

    def decompose(self, nl_description: str) -> Scenario:
        system_prompt = LoadPrompt.load_prompt(
            "gherkin_decomposition.jinja2", prompt_format=PromptFormat.JINJA2, prompt_type="system"
        ).format()
        chat_prompt = LoadPrompt.load_prompt(
            "gherkin_decomposition.jinja2", prompt_format=PromptFormat.JINJA2, prompt_type="chat"
        ).format(input=nl_description)

        scenario: Scenario = self.structured.invoke_prompts(
            system=system_prompt,
            chat=chat_prompt,
            schema=Scenario,
            strict=True,
        )

        return scenario

