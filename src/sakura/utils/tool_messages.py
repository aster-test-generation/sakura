from __future__ import annotations
_B='status'
_A=None
import json
from typing import Any,Dict,Mapping
def build_tool_ok(data:Any)->Dict[str,Any]:'Construct a standard success envelope for tool outputs.';return{_B:'ok','data':to_json_safe(data)}
def build_tool_error(code:str,message:str,*,details:Mapping[str,Any]|_A=_A)->Dict[str,Any]:'Construct a standard error envelope for tool outputs.';B='error';A=details;return{_B:B,B:{'code':code,'message':message,'details':to_json_safe(dict(A)if A is not _A else{})}}
def dumps_envelope(envelope:Dict[str,Any])->str:'Serialize an envelope with consistent JSON formatting.';return json.dumps(envelope,separators=(',',':'),ensure_ascii=False)
def format_tool_ok(data:Any)->str:'Serialize a success envelope.';return dumps_envelope(build_tool_ok(data))
def format_tool_error(code:str,message:str,*,details:Mapping[str,Any]|_A=_A)->str:'Serialize an error envelope.';return dumps_envelope(build_tool_error(code,message,details=details))
def to_json_safe(value:Any)->Any:
	'Convert values to JSON-serializable structures, falling back to string.';A=value
	if hasattr(A,'model_dump'):
		try:return to_json_safe(A.model_dump())
		except Exception:return str(A)
	if isinstance(A,dict):return{str(A):to_json_safe(B)for(A,B)in A.items()}
	if isinstance(A,(list,tuple,set)):B=[to_json_safe(A)for A in A];return B if isinstance(A,list)else B
	try:json.dumps(A);return A
	except TypeError:return str(A)