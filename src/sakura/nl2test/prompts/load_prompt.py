from enum import Enum
from pathlib import Path
from typing import List, Literal

from langchain_core.prompts import PromptTemplate


class PromptFormat(Enum):
    JINJA2 = "jinja2"


class LoadPrompt:
    @staticmethod
    def load_prompt(
        file_name: str,
        prompt_format: PromptFormat,
        prompt_type: Literal["chat", "system"],
    ) -> PromptTemplate:
        prompt_file = Path(__file__).parent / "templates" / prompt_type / file_name

        try:
            template_str = prompt_file.read_text()
        except Exception as exc:
            raise FileNotFoundError(f"File {prompt_file} not found") from exc

        prompt_template = PromptTemplate.from_template(
            template_str,
            template_format=prompt_format.value,
        )
        return prompt_template
