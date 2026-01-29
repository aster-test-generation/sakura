from enum import Enum
from pathlib import Path
from typing import List,Literal
from langchain_core.prompts import PromptTemplate
class PromptFormat(Enum):JINJA2='jinja2'
class LoadPrompt:
	@staticmethod
	def load_prompt(file_name:str,prompt_format:PromptFormat,prompt_type:Literal['chat','system'])->PromptTemplate:
		A=Path(__file__).parent/'templates'/prompt_type/file_name
		try:B=A.read_text()
		except Exception as C:raise FileNotFoundError(f"File {A} not found")from C
		D=PromptTemplate.from_template(B,template_format=prompt_format.value);return D