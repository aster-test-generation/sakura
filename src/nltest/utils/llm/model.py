_A=None
from dataclasses import dataclass,field
from enum import Enum
from typing import Optional,Dict,Any
class Provider(Enum):OPENROUTER='openrouter';VLLM='vllm';OLLAMA='ollama';OPENAI='openai';GCP='gcp';MISTRAL='mistral'
@dataclass
class LLMSettings:provider:Provider;model:str;client_type:Optional['ClientType']=_A;temperature:float=-1;max_tokens:Optional[int]=_A;base_url:Optional[str]=_A;api_key:Optional[str]=_A;request_timeout:int=120;default_headers:Dict[str,str]=field(default_factory=dict);model_kwargs:Dict[str,Any]=field(default_factory=dict)
class ClientType(Enum):CODE_GEN='code_gen';SUMMARIZATION='summarization';DECISION='decision';STRUCTURED='structured'