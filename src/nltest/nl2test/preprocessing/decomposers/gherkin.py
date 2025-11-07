from __future__ import annotations

from nltest.nl2test.models.decomposition import Scenario
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.nl2test.preprocessing.decomposers.base import BaseDecomposer
from nltest.utils.llm import LLMClient, ClientType, UsageTracker


class GherkinDecomposer(BaseDecomposer):
    def __init__(self, *, usage_tracker: UsageTracker | None = None) -> None:
        tracker = usage_tracker or UsageTracker()
        self.structured = LLMClient(
            ClientType.STRUCTURED,
            usage_tracker=tracker,
        )
        self.usage_tracker = tracker

    def decompose(self, nl_description: str) -> Scenario:
        system_prompt = LoadPrompt.load_prompt(
            "gherkin_decomposition.jinja2", prompt_format=PromptFormat.JINJA2, prompt_type="system"
        ).format()
        chat_prompt = LoadPrompt.load_prompt(
            "gherkin_decomposition.jinja2", prompt_format=PromptFormat.JINJA2, prompt_type="chat"
        ).format(input=nl_description)

        scenario: Scenario = self.invoke_with_retries(
            client=self.structured,
            system_prompt=system_prompt,
            base_chat_prompt=chat_prompt,
            schema=Scenario,
            strict=True,
        )

        return scenario
