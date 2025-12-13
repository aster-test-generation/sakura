from enum import Enum
from pathlib import Path
from typing import Literal

from jinja2 import Environment, FileSystemLoader, Template
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
        except:
            raise FileNotFoundError(f"File {prompt_file} not found")

        prompt_template = PromptTemplate.from_template(
            template_str,
            template_format=prompt_format.value,
        )
        return prompt_template

    @staticmethod
    def load_jinja2_template(
        file_name: str, prompt_type: Literal["chat", "system"]
    ) -> Template:
        """
        Load a Jinja2 template using standard Jinja2 (needed for loop.index) since LangChain blocks attribute access in templates.
        """
        template_dir = Path(__file__).parent / "templates" / prompt_type
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        return env.get_template(file_name)
