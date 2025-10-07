from __future__ import annotations

from typing import List

from nltest.nl2test.models.decomposition import GrammaticalBlock, GrammaticalBlockList
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.nl2test.preprocessing.decomposers.base import BaseDecomposer
from nltest.utils.llm import LLMClient, ClientType


class GrammaticalDecomposer(BaseDecomposer):
    def __init__(self) -> None:
        self.structured = LLMClient(ClientType.STRUCTURED)

    def decompose(self, nl_description: str) -> GrammaticalBlockList:
        system_prompt = LoadPrompt.load_prompt(
            "grammatical_decomposition.jinja2", prompt_format=PromptFormat.JINJA2, prompt_type="system"
        ).format()
        chat_prompt = LoadPrompt.load_prompt(
            "grammatical_decomposition.jinja2", prompt_format=PromptFormat.JINJA2, prompt_type="chat"
        ).format(input=nl_description)

        result: GrammaticalBlockList = self.structured.invoke_prompts(
            system=system_prompt,
            chat=chat_prompt,
            schema=GrammaticalBlockList,
            strict=True,
        )

        return result
