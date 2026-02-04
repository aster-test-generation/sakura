from __future__ import annotations
from sakura.nl2test.models.decomposition import GrammaticalBlockList
from sakura.nl2test.prompts.load_prompt import LoadPrompt,PromptFormat
from sakura.nl2test.preprocessing.decomposers.base import BaseDecomposer
from sakura.utils.llm import LLMClient,ClientType,UsageTracker
class GrammaticalDecomposer(BaseDecomposer):
	def __init__(A,*,usage_tracker:UsageTracker|None=None)->None:B=usage_tracker or UsageTracker();A.structured=LLMClient(ClientType.STRUCTURED,usage_tracker=B);A.usage_tracker=B
	def decompose(A,nl_description:str)->GrammaticalBlockList:B='grammatical_decomposition.jinja2';C=LoadPrompt.load_prompt(B,prompt_format=PromptFormat.JINJA2,prompt_type='system').format();D=LoadPrompt.load_prompt(B,prompt_format=PromptFormat.JINJA2,prompt_type='chat').format(input=nl_description);E:GrammaticalBlockList=A.invoke_with_retries(client=A.structured,system_prompt=C,base_chat_prompt=D,schema=GrammaticalBlockList,strict=True);return E