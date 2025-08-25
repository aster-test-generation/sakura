import json
import re
from typing import List

from pydantic import RootModel, BaseModel, Field

from nltest.nl2test.model.models import GrammaticalBlock
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.utils.llm.llm_client import LLMClient, ClientType
from nltest.utils.pretty.prints import pretty_print
from langchain_core.messages import SystemMessage, HumanMessage


class GrammaticalBlockList(BaseModel):
    blocks: List[GrammaticalBlock] = Field(..., description="Ordered list of grammatical blocks.")


class NLDecomposer:
    def __init__(self):
        self.structured = LLMClient(ClientType.STRUCTURED)

    def decompose(self, nl_description: str) -> List[GrammaticalBlock]:
        system_prompt = LoadPrompt.load_prompt(
            "nl_decomposition.jinja2", prompt_format=PromptFormat.JINJA2, prompt_type="system"
        ).format()
        chat_prompt = LoadPrompt.load_prompt(
            "nl_decomposition.jinja2", prompt_format=PromptFormat.JINJA2, prompt_type="chat"
        ).format(input=nl_description)

        result: GrammaticalBlockList = self.structured.invoke_prompts(
            system=system_prompt,
            chat=chat_prompt,
            schema=GrammaticalBlockList,
            strict=True,
        )

        return result.blocks
