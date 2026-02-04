from __future__ import annotations
_F='status'
_E='name'
_D='str'
_C=False
_B=True
_A=None
import json
from typing import Any,Dict,List,Optional,Set,Tuple
from langchain_core.messages import AIMessage,ToolMessage
from sakura.nl2test.models import AgentState
class MessageRedactor:
	'\n    Shared utility for redacting stale tool outputs and inputs from agent message history.\n    Used to reduce token usage by replacing outdated data with placeholders.\n    '
	@staticmethod
	def _load_payload(content:Any)->Tuple[Optional[Dict[str,Any]],str]:
		'Load JSON payload from message content, returning (payload, type_hint).';A=content
		if isinstance(A,str):
			try:return json.loads(A),_D
			except json.JSONDecodeError:return _A,_D
		if isinstance(A,dict):return A,'dict'
		return _A,'other'
	@staticmethod
	def _store_payload(message:ToolMessage,payload:Dict[str,Any],payload_type:str)->_A:
		'Store payload back into message content with appropriate format.';B=payload;A=message
		if payload_type==_D:A.content=json.dumps(B,separators=(',',':'),ensure_ascii=_C)
		else:A.content=B
	@staticmethod
	def _build_call_lookup(state:AgentState)->Dict[str,str]:
		'Build mapping from tool_call_id to tool_name from AIMessages.';A:Dict[str,str]={}
		for B in state.messages:
			if isinstance(B,AIMessage):
				F=B.tool_calls or[]
				for C in F:
					D=C.get('id');E=C.get(_E)
					if D and E:A[D]=E
		return A
	@staticmethod
	def redact_tool_outputs(state:AgentState,tool_names:Set[str],keys_to_redact:Dict[str,str],status_filter:str='ok')->_A:
		'\n        Redact specific keys in tool output messages for given tool names.\n\n        Args:\n            state: The agent state containing messages to redact\n            tool_names: Set of tool names whose outputs should be redacted\n            keys_to_redact: Dict mapping data keys to their placeholder strings\n            status_filter: Only redact messages with this status (default "ok")\n        ';E=state;I=MessageRedactor._build_call_lookup(E)
		for A in E.messages:
			if not isinstance(A,ToolMessage):continue
			F=A.tool_call_id;C=A.name
			if not C and F:C=I.get(F)
			if C not in tool_names:continue
			B,J=MessageRedactor._load_payload(A.content)
			if not isinstance(B,dict):continue
			if B.get(_F)!=status_filter:continue
			D=B.get('data')
			if not isinstance(D,dict):continue
			G=_C
			for(H,K)in keys_to_redact.items():
				if H in D:D[H]=K;G=_B
			if G:MessageRedactor._store_payload(A,B,J)
	@staticmethod
	def redact_tool_outputs_by_tool(state:AgentState,tool_redactions:Dict[str,Dict[str,str]],status_filter:str='ok')->_A:
		'\n        Redact tool outputs with tool-specific key mappings.\n\n        Args:\n            state: The agent state containing messages to redact\n            tool_redactions: Dict mapping tool_name -> {key: placeholder}\n            status_filter: Only redact messages with this status (default "ok")\n        ';F=tool_redactions;E=state;J=MessageRedactor._build_call_lookup(E)
		for A in E.messages:
			if not isinstance(A,ToolMessage):continue
			G=A.tool_call_id;B=A.name
			if not B and G:B=J.get(G)
			if B not in F:continue
			C,K=MessageRedactor._load_payload(A.content)
			if not isinstance(C,dict):continue
			if C.get(_F)!=status_filter:continue
			D=C.get('data')
			if not isinstance(D,dict):continue
			L=F[B];H=_C
			for(I,M)in L.items():
				if I in D:D[I]=M;H=_B
			if H:MessageRedactor._store_payload(A,C,K)
	@staticmethod
	def redact_tool_inputs(state:AgentState,tool_name:str,keys_to_redact:Dict[str,str],llm_client:Any,exclude_latest:bool=_B)->_A:
		"\n        Redact specific keys in tool call arguments for a given tool name.\n\n        Args:\n            state: The agent state containing messages to redact\n            tool_name: Name of the tool whose inputs should be redacted\n            keys_to_redact: Dict mapping argument keys to their placeholder strings\n            llm_client: LLM client with parse_tool_args method\n            exclude_latest: If True, don't redact the most recent call (default True)\n        ";L=exclude_latest;K=tool_name;J=state;F='args';G:Optional[str]=_A
		if L:
			for B in reversed(J.messages):
				if not isinstance(B,AIMessage):continue
				D=B.tool_calls or[]
				for A in reversed(D):
					if A.get(_E)==K:G=A.get('id');break
				if G:break
		for B in J.messages:
			if not isinstance(B,AIMessage):continue
			D=B.tool_calls or[]
			for(H,A)in enumerate(D):
				if A.get(_E)!=K:continue
				if L and A.get('id')==G:continue
				E=A.get(F);M=llm_client.parse_tool_args(E)
				if isinstance(M,dict):C=dict(M)
				elif isinstance(E,dict):C=dict(E)
				else:C={}
				N=_C
				for(O,Q)in keys_to_redact.items():
					if O in C:C[O]=Q;N=_B
				if not N:continue
				try:
					if isinstance(E,str):A[F]=json.dumps(C,ensure_ascii=_B,sort_keys=_B)
					else:A[F]=C
				except Exception:A[F]=C
				D[H]=A;P=B.additional_kwargs
				if isinstance(P,dict):
					I=P.get('tool_calls')
					if isinstance(I,list)and H<len(I):I[H]=A
			B.tool_calls=D