import json
import re
from typing import List

from nltest.nl2test.model.models import GrammaticalBlock
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.utils.exceptions import FormatError
from nltest.utils.llm.format_validator import FormatValidator
from nltest.utils.llm.llm_client import LLMClient, ClientType
from nltest.utils.pretty.prints import pretty_print


class NLDecomposer:
    def __init__(self):
        self.llm = LLMClient(ClientType.STRUCTURED)

    def decompose(self, nl_description: str) -> List[GrammaticalBlock]:
        prompt = LoadPrompt().load_prompt("nl_decomposition.jinja2", prompt_format=PromptFormat.JINJA2)
        prompt = prompt.format(input=nl_description)
        output = self.llm.generate(prompt, sanitize=True)

        try:
            blocks: List[GrammaticalBlock] = FormatValidator.validate(output, List[GrammaticalBlock])
        except ValueError as e:
            raise FormatError("Formatting failed", extra_info={"raw_output": output, "error": str(e)})
        # TODO: Decide if we want to use LLM for reparsing

        return blocks



