from __future__ import annotations
from abc import ABC,abstractmethod
from typing import Any,Type,TypeVar
from pydantic import BaseModel
from sakura.utils.llm import LLMClient
SchemaT=TypeVar('SchemaT',bound=BaseModel)
class BaseDecomposer(ABC):
	@abstractmethod
	def decompose(self,nl_description:str)->Any:raise NotImplementedError
	def invoke_with_retries(B,*,client:LLMClient,system_prompt:str,base_chat_prompt:str,schema:Type[SchemaT],strict:bool=True,total_attempts:int=3)->SchemaT:'Call structured LLM output with retries and feedback about prior failures.';A=client.invoke_structured_with_retries(system=system_prompt,chat=base_chat_prompt,schema=schema,strict=strict,max_attempts=total_attempts,on_failure='raise');assert A is not None;return A