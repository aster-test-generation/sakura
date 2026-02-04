from __future__ import annotations
_L='return_none'
_K='usage_metadata'
_J='content'
_I='json_mode'
_H='function_calling'
_G='response'
_F=False
_E='tool_calls'
_D='provider'
_C='json_schema'
_B=True
_A=None
import json,re,textwrap,traceback,uuid
from typing import Any,Dict,Literal,Optional,Sequence,Type,TypeVar,Union
from langchain_core.messages import AIMessage,BaseMessage,HumanMessage,SystemMessage,ToolMessage
from langchain_core.runnables import RunnableSerializable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from openai import LengthFinishReasonError
from pydantic import BaseModel,SecretStr
from tenacity import RetryCallState,retry,retry_if_exception,stop_after_attempt,wait_chain,wait_random
from sakura.utils.pretty.color_logger import RichLog
from..config.config import Config
from..exceptions import ConfigurationException
from.model import ClientType,Provider
from.usage_tracker import UsageTracker
from.general_prompts.structured_retry import LENGTH_EXCEEDED_RETRY_PROMPT,STRUCTURED_OUTPUT_RETRY_PROMPT
SchemaT=TypeVar('SchemaT',bound=BaseModel)
class EmptyLLMResponseError(RuntimeError):'Raised when the provider returns an AIMessage with no content and no tool calls.'
def _is_retriable_error(exc:BaseException)->bool:
	'Check if exception is transient and worth retrying.';C='status_code';A=exc
	if isinstance(A,EmptyLLMResponseError):return _B
	D=getattr(A,C,_A)or getattr(getattr(A,_G,_A),C,_A)
	if D in{429,500,502,503,504}:return _B
	if isinstance(A,(ConnectionError,TimeoutError)):return _B
	B=str(A).lower()
	if'rate limit'in B or'too many requests'in B or'overloaded'in B:return _B
	if isinstance(A,ValueError)and"does not have a 'parsed' field nor a 'refusal' field"in str(A):return _B
	return _F
def _log_retry_attempt(retry_state:RetryCallState)->_A:'Log retry attempts for observability.';A=retry_state;C=A.attempt_number;B=A.outcome.exception()if A.outcome else _A;D=A.next_action.sleep if A.next_action else 0;RichLog.warn(f"[LLMClient] Retry attempt {C} after {D:.1f}s due to: {type(B).__name__ if B else'unknown'}")
class LLMClient:
	def __init__(C,client_type:ClientType,*,usage_tracker:UsageTracker|_A=_A):
		X='parallel_tool_calls';W='exclude';V='effort';N=client_type;M='max_tokens';I='reasoning';B='llm';A=Config();H=A.get(B,'api_url');J=A.get(B,_D)
		if J in(_A,''):F=_A
		else:
			try:F=Provider(J)
			except ValueError as Y:raise ConfigurationException(f"Invalid LLM provider: {J}. Must be one of {list(Provider)}")from Y
		if F is _A and not H:raise ConfigurationException('LLM provider is not configured and no API URL was supplied.')
		K=A.get(B,'model');Z=A.get(B,f"{N.value}_temp");a=A.get(B,'api_key')
		try:L=A.get(B,M)
		except ConfigurationException:L=16384
		try:O=A.get(B,'timeout')
		except ConfigurationException:O=_A
		try:D=A.get(B,'default_headers')
		except ConfigurationException:D=_A
		try:G=A.get(B,'model_kwargs')
		except ConfigurationException:G={}
		E:dict[str,Any]=G.pop('extra_body',{})
		try:P=A.get(I,'configure')
		except ConfigurationException:P=_F
		if F==Provider.OPENROUTER and P:
			try:Q=A.get(I,V)
			except ConfigurationException:Q='low'
			try:R=A.get(I,W)
			except ConfigurationException:R=_F
			E[I]={V:Q,W:R}
		if F==Provider.OPENROUTER:
			D={}if D is _A else D;D.setdefault('HTTP-Referer','http://localhost');D.setdefault('X-Title','NL2Test LLM Client')
			try:
				S=A.get('openrouter','ignore_providers')
				if S:
					if _D not in E:E[_D]={}
					E[_D]['ignore']=S
			except ConfigurationException:pass
		try:T=A.get(B,'can_parallel_tool')
		except ConfigurationException:T=_F
		if X not in G:G[X]=bool(T)
		if F==Provider.MISTRAL:E[M]=L;U={}
		else:U={M:L}
		C._chat=ChatOpenAI(model=K,temperature=Z,**U,base_url=H.rstrip('/')if H else _A,api_key=SecretStr(a),timeout=O,default_headers=D,model_kwargs=G,extra_body=E if E else _A)
		try:C._model_id=C._chat.model_name
		except Exception:C._model_id=K
		C._usage_tracker=usage_tracker or UsageTracker();C._provider=F;C._model=K;C._base_url=H;C._client_type=N
	def _build_runnable(G,*,tools:Optional[Sequence[BaseTool]]=_A,tool_choice:Union[str,dict,_A]='auto',response_format:Optional[Dict[str,Any]]=_A,extra_model_kwargs:Optional[Dict[str,Any]]=_A,schema:Any=_A,strict:bool=_B,method:Optional[Literal[_C,_H,_I]]=_C,temperature:Optional[float]=_A)->RunnableSerializable:
		F=temperature;E=extra_model_kwargs;D=response_format;C=tools;B=schema;A:RunnableSerializable=G._chat
		if F is not _A:A=A.bind(temperature=F)
		if C:A=A.bind_tools(C,tool_choice=tool_choice)
		if D is not _A and B is _A:A=A.bind(response_format=D)
		if E:A=A.bind(**E)
		if B is not _A:A=A.with_structured_output(schema=B,strict=strict,method=method)
		return A
	@property
	def chat(self)->ChatOpenAI:return self._chat
	def _normalize_tool_call_ids(E,ai_msg:AIMessage)->AIMessage:
		'Ensure tool call IDs are always present by generating unique IDs if missing.';A=ai_msg
		if getattr(A,_E,_A):
			C=[]
			for B in A.tool_calls:D=B.get('id')or f"tool_{uuid.uuid4().hex[:8]}";B['id']=D;C.append(B)
			A.tool_calls=C
		return A
	@staticmethod
	def _message_preview(message:BaseMessage|_A,limit:int=500)->str:
		E=message
		if E is _A:return''
		A=getattr(E,_J,'')
		if isinstance(A,str):B=A
		elif isinstance(A,list):
			D=[]
			for C in A:
				if isinstance(C,dict):D.append(str(C.get('text')or C.get(_J)or C))
				else:D.append(str(C))
			B='\n'.join(D)
		else:B=str(A)
		B=B.strip();return B[:limit]
	def _summarize_messages(C,messages:Sequence[BaseMessage])->Dict[str,Any]:
		E='types';B=messages;A:Dict[str,Any]={'count':len(B),E:{}}
		for G in B:F=type(G).__name__;A[E][F]=A[E].get(F,0)+1
		H=next((A for A in reversed(B)if isinstance(A,SystemMessage)),_A);I=next((A for A in reversed(B)if isinstance(A,HumanMessage)),_A);J=next((A for A in reversed(B)if isinstance(A,ToolMessage)),_A);D=next((A for A in reversed(B)if isinstance(A,AIMessage)),_A);A['last_message_type']=type(B[-1]).__name__ if B else _A;A['system_preview']=C._message_preview(H);A['last_human_preview']=C._message_preview(I);A['last_tool_preview']=C._message_preview(J);A['last_ai_preview']=C._message_preview(D)
		if D is not _A:K=getattr(D,_E,_A)or[];A['last_ai_tool_calls']=[A.get('name')for A in K if isinstance(A,dict)]
		return A
	def _log_empty_response(A,out:AIMessage,messages:Sequence[BaseMessage],context:Optional[Dict[str,Any]]=_A)->_A:
		C='response_metadata';B=out
		def D(value:Any)->int:
			A=value
			if isinstance(A,str):return len(A)
			if isinstance(A,list):return len(A)
			if A is _A:return 0
			return len(str(A))
		E={_D:getattr(A._provider,'value',A._provider),'model':A._model,'base_url':A._base_url,'client':A._client_type.value,'context':context or{},_G:{'content_type':type(B.content).__name__,'content_len':D(B.content),_E:getattr(B,_E,_A),_K:getattr(B,_K,_A),C:getattr(B,C,_A)},'messages':A._summarize_messages(messages)};RichLog.error(f"[LLMClient] empty_response details={json.dumps(E,default=str)[:4000]}")
	@retry(stop=stop_after_attempt(5),wait=wait_chain(wait_random(1,2),wait_random(5,10),wait_random(20,30),wait_random(45,60)),retry=retry_if_exception(_is_retriable_error),before_sleep=_log_retry_attempt,reraise=_B)
	def _invoke_with_retry(self,runnable:RunnableSerializable,messages:Sequence[BaseMessage],context:Optional[Dict[str,Any]]=_A)->Any:
		'Internal method that performs the actual invocation with retry logic.';C=messages;A=runnable.invoke(list(C))
		if isinstance(A,AIMessage):
			D=bool(getattr(A,_E,_A))
			if isinstance(A.content,str):B=bool(A.content.strip())
			elif isinstance(A.content,list):B=len(A.content)>0
			else:B=bool(A.content)
			if not D and not B:self._log_empty_response(A,C,context);raise EmptyLLMResponseError('LLM returned an empty AIMessage (no content, no tool_calls).')
		return A
	def invoke_messages(A,messages:Sequence[BaseMessage],*,tools:Optional[Sequence[BaseTool]]=_A,tool_choice:Union[str,dict,_A]='auto',response_format:Optional[Dict[str,Any]]=_A,extra_model_kwargs:Optional[Dict[str,Any]]=_A,schema:Any=_A,strict:bool=_B,method:Optional[Literal[_C,_H,_I]]=_C,temperature:Optional[float]=_A,context:Optional[Dict[str,Any]]=_A)->Any:
		J='yes';H=schema;G=extra_model_kwargs;F=response_format;E=tool_choice;D=messages;C=context;K=A._build_runnable(tools=tools,tool_choice=E,response_format=F,extra_model_kwargs=G,schema=H,strict=strict,method=method,temperature=temperature)
		try:B=A._invoke_with_retry(K,D,C)
		except Exception as I:
			L=type(I).__name__;RichLog.error(f"[LLMClient] {L} during invoke (provider={getattr(A._provider,'value',A._provider)}, model={A._model}, base_url={A._base_url}, client={A._client_type.value}): {I}");RichLog.debug(traceback.format_exc())
			if C:RichLog.debug(f"context: {json.dumps(C,default=str)[:2000]}")
			RichLog.debug(f"opts: tool_choice={E}, schema={J if H is not _A else'no'}, response_format={J if F is not _A else'no'}, extra_model_kwargs={str(G)[:500]}");M=[type(A).__name__ for A in D];RichLog.debug(f"messages: {','.join(M)}");raise
		if hasattr(B,_K)and B.usage_metadata:N=B.usage_metadata.get('output_tokens',0);O=B.usage_metadata.get('output_token_details')or{};P=O.get('reasoning_tokens',0);A._usage_tracker.record(input_tokens=B.usage_metadata.get('input_tokens',0),output_tokens=N+P)
		return A._normalize_tool_call_ids(B)if isinstance(B,AIMessage)else B
	def invoke_prompts(A,system:str,chat:str,*,response_format:Optional[Dict[str,Any]]=_A,extra_model_kwargs:Optional[Dict[str,Any]]=_A,schema:Any=_A,strict:bool=_B,method:Optional[Literal[_C,_H,_I]]=_C,temperature:Optional[float]=_A)->Any:B:Sequence[BaseMessage]=[SystemMessage(content=system),HumanMessage(content=chat)];return A.invoke_messages(B,response_format=response_format,extra_model_kwargs=extra_model_kwargs,schema=schema,strict=strict,method=method,temperature=temperature)
	def invoke_structured_with_retries(V,*,messages:Sequence[BaseMessage]|_A=_A,system:str|_A=_A,chat:str|_A=_A,schema:Type[SchemaT],strict:bool=_B,max_attempts:int=3,retry_prompt_template:str|_A=_A,on_failure:Literal['raise',_L]='raise')->SchemaT|_A:
		'\n        Invoke LLM with structured output binding, retrying on validation failures.\n\n        Accepts EITHER:\n        - messages: A pre-built message list (must end with HumanMessage)\n        - system + chat: Simple prompt pair (for decomposer use cases)\n\n        On failure, modifies the last HumanMessage to include retry context.\n        ';E=chat;D=system;C=messages
		if C is not _A and(D is not _A or E is not _A):raise ValueError("Provide either 'messages' OR 'system'+'chat', not both.")
		if C is _A and(D is _A or E is _A):raise ValueError("Must provide either 'messages' or both 'system' and 'chat'.")
		W=retry_prompt_template or STRUCTURED_OUTPUT_RETRY_PROMPT;O=C is not _A;A:list[BaseMessage];F:str
		if O:
			A=list(C)
			if not A or not isinstance(A[-1],HumanMessage):raise ValueError('When using messages mode, the last message must be a HumanMessage.')
			I=A[-1];F=I.content if isinstance(I.content,str)else str(I.content)
		else:assert D is not _A and E is not _A;A=[SystemMessage(content=D)];F=E
		G:list[str]=[];H:Exception|_A=_A
		for P in range(1,max_attempts+1):
			J=list(A)
			if G:X='\n\n'.join(G);K=f"{F}\n\n{X}"
			else:K=F
			if O:J[-1]=HumanMessage(content=K)
			else:J.append(HumanMessage(content=K))
			try:Y=V.invoke_messages(J,schema=schema,strict=strict);return Y
			except LengthFinishReasonError as Q:
				H=Q;L='';R=getattr(Q,'completion',_A)
				if R:
					S=getattr(R,'choices',[])
					if S:
						T=getattr(S[0],'message',_A)
						if T:L=getattr(T,_J,'')or''
				M=LENGTH_EXCEEDED_RETRY_PROMPT.format(attempt=P,partial_output=L[:500]if L else'(none available)');G.append(M)
			except Exception as B:
				if _is_retriable_error(B):raise
				H=B;Z=(str(B)or repr(B)).strip()[:800];N=getattr(B,'raw_output',_A)
				if N is _A:
					U=getattr(B,_G,_A)
					if U is not _A:N=getattr(U,'text',_A)
				a=str(N or'')[:800];M=W.format(attempt=P,error_excerpt=Z,output_excerpt=a);G.append(M)
		if on_failure==_L:return
		if H is not _A:raise H
		raise RuntimeError('Structured prompt invocation failed without raising an error.')
	@staticmethod
	def sanitize(text:str)->str:
		A=text;B=[lambda t:re.sub('(?si).*?</think>','',t,flags=re.IGNORECASE)]
		for C in B:A=C(A)
		return A.strip()
	@staticmethod
	def _normalize_string_values(value:Any)->Any:
		'Recursively dedent and strip string values to normalize LLM tool args.';A=value
		if isinstance(A,str):return textwrap.dedent(A).strip()
		if isinstance(A,dict):return{A:LLMClient._normalize_string_values(B)for(A,B)in A.items()}
		if isinstance(A,list):return[LLMClient._normalize_string_values(A)for A in A]
		if isinstance(A,tuple):return tuple(LLMClient._normalize_string_values(A)for A in A)
		return A
	@staticmethod
	def parse_tool_args(args:Union[str,Dict[str,Any]])->Dict[str,Any]:
		'Tool call args can be passed as a JSON string or a dict.';A=args
		if isinstance(A,dict):return LLMClient._normalize_string_values(A)
		if isinstance(A,str):
			try:B=json.loads(A)
			except Exception:return{}
			if not isinstance(B,dict):return{}
			return LLMClient._normalize_string_values(B)
		return{}