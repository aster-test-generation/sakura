from __future__ import annotations

from nltest.nl2test.models.decomposition import GrammaticalBlockList
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.nl2test.preprocessing.decomposers.base import BaseDecomposer
from nltest.utils.llm import LLMClient, ClientType, UsageTracker


class GrammaticalDecomposer(BaseDecomposer):
    def __init__(self, *, usage_tracker: UsageTracker | None = None) -> None:
        tracker = usage_tracker or UsageTracker()
        self.structured = LLMClient(
            ClientType.STRUCTURED,
            usage_tracker=tracker,
        )
        self.usage_tracker = tracker

    def decompose(self, nl_description: str) -> GrammaticalBlockList:
        system_prompt = LoadPrompt.load_prompt(
            "grammatical_decomposition.jinja2", prompt_format=PromptFormat.JINJA2, prompt_type="system"
        ).format()
        chat_prompt = LoadPrompt.load_prompt(
            "grammatical_decomposition.jinja2", prompt_format=PromptFormat.JINJA2, prompt_type="chat"
        ).format(input=nl_description)

        result: GrammaticalBlockList = self.invoke_with_retries(
            client=self.structured,
            system_prompt=system_prompt,
            base_chat_prompt=chat_prompt,
            schema=GrammaticalBlockList,
            strict=True,
        )

        return result
