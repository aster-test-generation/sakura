_B='templates'
_A='system'
from enum import Enum
from pathlib import Path
from typing import Literal
from jinja2 import Environment,FileSystemLoader,Template
from langchain_core.prompts import PromptTemplate
class PromptFormat(Enum):JINJA2='jinja2'
class LoadPrompt:
	@staticmethod
	def load_prompt(file_name:str,prompt_format:PromptFormat,prompt_type:Literal['chat',_A])->PromptTemplate:
		A=Path(__file__).parent/_B/prompt_type/file_name
		try:B=A.read_text()
		except:raise FileNotFoundError(f"File {A} not found")
		C=PromptTemplate.from_template(B,template_format=prompt_format.value);return C
	@staticmethod
	def load_jinja2_template(file_name:str,prompt_type:Literal['chat',_A])->Template:'\n        Load a Jinja2 template using standard Jinja2 (needed for loop.index) since LangChain blocks attribute access in templates.\n        ';A=Path(__file__).parent/_B/prompt_type;B=Environment(loader=FileSystemLoader(str(A)));return B.get_template(file_name)