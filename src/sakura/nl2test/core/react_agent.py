from __future__ import annotations
_N='message'
_M='details'
_L='status'
_K='Acknowledged.'
_J='finalize'
_I='error'
_H='args'
_G=False
_F='name'
_E='tool'
_D='tool_call_id'
_C=True
_B='id'
_A=None
import json
from typing import Any,Dict,List,Optional,Tuple,Type
from langchain_core.messages import AIMessage,BaseMessage,HumanMessage,SystemMessage,ToolCall,ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END,StateGraph
from pydantic import BaseModel
from sakura.nl2test.models import AgentState
from sakura.utils.exceptions import ConfigurationException
from sakura.utils.llm import LLMClient
from sakura.utils.pretty.color_logger import RichLog
from sakura.utils.tool_messages import format_tool_error,format_tool_ok
class ReActAgent:
	def __init__(A,*,llm:LLMClient,tools:List[BaseTool],allow_duplicate_tools:Optional[List[BaseTool]]=_A,system_message:Optional[str]=_A,allow_parallelize:bool=_C,max_iters:int=20,strict_finalize:bool=_C,use_checkpointer:bool=_C,max_force_end_attempts:int=3,max_no_tool_retries:int=2)->_A:
		B=tools;(A.llm):LLMClient=llm;A.tools=B;A.allow_duplicate_tools=allow_duplicate_tools or[];A.max_iters=max_iters;A.system_message=system_message or'You are a helpful AI assistant.';A.allow_parallelize=allow_parallelize;A.strict_finalize=strict_finalize;A.use_checkpointer=use_checkpointer;A.max_force_end_attempts=max_force_end_attempts;A.max_no_tool_retries=max_no_tool_retries;A._allow_duplicate_tool_names={A.name for A in A.allow_duplicate_tools};(A.tool_map):Dict[str,BaseTool]={A.name:A for A in B};(A._end_now):bool=_G
		if A.strict_finalize:
			C=any(A.name==_J for A in A.tools)
			if not C:raise ConfigurationException('finalize_tool_missing',message="ReActAgent requires a 'finalize' tool when strict_finalize is enabled.")
		A.graph=A._build_graph()
	def reset_agent(A)->_A:'Reset internal termination flag so the agent can continue running.';A._end_now=_G
	def _get_finalize_schema(A)->Type[BaseModel]:'Return the Pydantic schema for finalize args. Must be implemented by subclasses.';raise NotImplementedError('Subclasses must implement _get_finalize_schema() to return a Pydantic schema.')
	def _get_force_finalize_system_prompt(A)->str:'Return system prompt for force_finalize structured output. Must be implemented by subclasses.';raise NotImplementedError('Subclasses must implement _get_force_finalize_system_prompt() to return a system prompt.')
	def _get_force_finalize_chat_prompt(A)->str:'Return chat prompt for force_finalize structured output. Must be implemented by subclasses.';raise NotImplementedError('Subclasses must implement _get_force_finalize_chat_prompt() to return a chat prompt.')
	def _process_force_finalize_result(A,result:BaseModel,state:AgentState)->_A:'Process the structured finalize result. Must be implemented by subclasses.';raise NotImplementedError('Subclasses must implement _process_force_finalize_result() to process the result.')
	def _execute_force_end(A,state:AgentState)->AgentState:
		'Execute force finalize logic. Override in subclasses to customize behavior.';B=state;B.force_end_attempts+=1;F=A._get_finalize_schema();G=A._get_force_finalize_system_prompt();H=A._get_force_finalize_chat_prompt();C:List[BaseMessage]=[SystemMessage(content=G)]
		for D in B.messages:
			if isinstance(D,BaseMessage)and not isinstance(D,SystemMessage):C.append(D)
		if C and isinstance(C[-1],ToolMessage):C.append(AIMessage(content=_K))
		C.append(HumanMessage(content=H))
		try:E=A.llm.invoke_structured_with_retries(messages=C,schema=F,strict=_C,max_attempts=A.max_force_end_attempts,on_failure='return_none')
		except Exception as I:RichLog.warn(f"[ReActAgent] force_end structured output failed: {I}");E=_A
		if E is not _A:A._process_force_finalize_result(E,B);A._log_force_finalize(B);return B
		if A.strict_finalize:raise ConfigurationException('finalize_not_called',message='Failed to produce finalize output after max attempts.',details={'attempts':A.max_force_end_attempts})
		B.finalize_called=_C;B.final_comments='Auto-finalized: structured output failed';A._log_force_finalize(B);return B
	def _log_force_finalize(D,state:AgentState)->_A:'Log finalize to tool trajectory and counts for force_end consistency.';C='<force_finalize>';A=state;A.curr_tool_trajectory.append(_J);B=A.total_tool_calls.setdefault(_J,{});B[C]=B.get(C,0)+1
	def prepare_tool_args(A,tool_name:str,raw_args:Dict[str,Any],state:AgentState)->Tuple[str,Dict[str,Any]]:'Hook for subclasses to inject or transform tool arguments before execution.';return tool_name,raw_args
	def _append_tool_error_if_needed(E,tool_call:ToolCall,result:Any,outputs:List[ToolMessage])->bool:
		B=result;A=tool_call
		if isinstance(B,dict)and B.get(_L)==_I:C=B.get(_I)or{};D=dict(C.get(_M)or{});D.setdefault(_E,A.get(_F));D.setdefault(_D,A.get(_B));outputs.append(ToolMessage(content=format_tool_error(code=C.get('code')or'tool_error',message=C.get(_N)or'Tool error',details=D),tool_call_id=A[_B]));return _C
		return _G
	def _append_tool_output(D,tool_call:ToolCall,result:Any,outputs:List[ToolMessage])->_A:
		C=outputs;B=result;A=tool_call
		if D._append_tool_error_if_needed(A,B,C):return
		C.append(ToolMessage(content=format_tool_ok(B),tool_call_id=A[_B]))
	def process_tool_output(D,tool_call:ToolCall,result:Any,state:AgentState,outputs:List[ToolMessage])->_A:
		'Allow subclasses to interpret tool results and update state.';B=outputs;A=tool_call
		try:D._append_tool_output(A,result,B)
		except Exception as C:B.append(ToolMessage(content=format_tool_error(code=type(C).__name__,message=str(C),details={_E:A.get(_F),_D:A[_B]}),tool_call_id=A[_B]))
	def process_llm_output(A,tool_call:ToolCall,state:AgentState)->_A:0
	def _should_end_after_tools(A,state:AgentState)->bool:'Hook for subclasses to request ending immediately after tools.';return A._end_now
	def _encode_tool_call(B,tool_name:str,args:Dict[str,Any])->str:
		try:A=json.dumps(args,sort_keys=_C,separators=(',',':'),ensure_ascii=_G)
		except Exception:A=str(args)
		return f"{tool_name}|{A}"
	def _log_tool_error(M,tool_msg:ToolMessage,llm_message:AIMessage)->_A:
		'Emit a RichLog debug entry containing the tool error and triggering LLM turn.';O=llm_message;N=tool_msg;E=N.content
		if isinstance(E,str):
			try:H=json.loads(E)
			except json.JSONDecodeError:return
		elif isinstance(E,dict):H=E
		else:return
		if H.get(_L)!=_I:return
		I=H.get(_I)or{};P=I.get(_M)or{};J=P.get(_E)or'unknown_tool';Q=P.get(_D)or getattr(N,_D,'unknown_call');U=I.get('code')or'unknown_error';V=I.get(_N)or'';F:ToolCall|_A=_A;W=O.tool_calls or[]
		for R in W:
			if R.get(_B)==Q:F=R;break
		if F:
			S='llm_tool_args';J=F.get(_F)or J;A=F.get(_H);K=M.llm.parse_tool_args(A)
			try:
				if K:B=json.dumps(K,ensure_ascii=_G)
				elif isinstance(A,str):B=A
				else:B=str(A)
			except Exception:B=str(K or A)
		else:
			S='llm_content';C=O.content
			if isinstance(C,str):G=C
			elif isinstance(C,list):
				L=[]
				for D in C:
					if isinstance(D,dict):L.append(str(D.get('text')or D.get('content')or D))
					else:L.append(str(D))
				G='\n'.join(L)
			else:G=str(C)
			try:T=M.llm.sanitize(G)
			except Exception:T=G
			B=T
		X=(B or'')[:500];RichLog.debug(f"[ReActAgent] tool_error name={J} call_id={Q} code={U} message={V} | {S}={X}")
	def _build_graph(A):
		'Organize the graph into nodes and edges. Typical react workflow with model call, tool call, and ending.';J='nudge';I='use_tools';H='continue';G='nudge_model';F='call_tools';E='end';D='call_model';C='force_end';B=StateGraph(AgentState)
		def K(state:AgentState)->AgentState:
			B=state;assert B.messages and isinstance(B.messages[0],SystemMessage),'First message must be a SystemMessage';F=A.max_iters-B.iterations;C=_A
			if F==1:C='SYSTEM NOTICE: Final iteration. Call the finalize tool now using the best available context.'
			elif F==2:C='SYSTEM NOTICE: Second-to-last iteration. Finish any remaining tool work now; plan to call finalize next turn.'
			D=list(B.messages)
			if C:
				E=D[-1]
				if isinstance(E,ToolMessage):D.append(AIMessage(content=_K));D.append(HumanMessage(content=C))
				elif isinstance(E,HumanMessage):G=E.content if isinstance(E.content,str)else str(E.content);D[-1]=HumanMessage(content=f"{G}\n\n{C}")
				else:D.append(HumanMessage(content=C))
			H:AIMessage=A.llm.invoke_messages(D,tools=A.tools,tool_choice='auto',extra_model_kwargs={'parallel_tool_calls':A.allow_parallelize},context={'agent':type(A).__name__,'iteration':B.iterations,'max_iters':A.max_iters,'no_tool_retries':B.no_tool_retries,'finalize_called':B.finalize_called});B.messages.append(H);B.iterations+=1;return B
		def L(state:AgentState)->AgentState:
			C=state;J:AIMessage=next(A for A in reversed(C.messages)if isinstance(A,AIMessage));E:List[ToolMessage]=[];F=J.tool_calls or[];Q=[A.get(_F)for A in F];RichLog.debug(f"[ReActAgent] call_tools: iteration={C.iterations}, tools={Q}");C.no_tool_retries=0;K:List[ToolCall]=[]
			if not A.allow_parallelize and len(F)>1:K=F[1:];F=F[:1]
			for D in F:
				B:str=D.get(_F);G:Dict[str,Any]=A.llm.parse_tool_args(D.get(_H));B,G=A.prepare_tool_args(B,G,C);L:BaseTool|_A=A.tool_map.get(B)
				if L is _A:E.append(ToolMessage(content=format_tool_error(code='tool_not_found',message=f"Tool '{B}' is not registered.",details={_E:B,_D:D[_B]}),tool_call_id=D[_B]));continue
				H=A._encode_tool_call(B,G);M=C.curr_tool_calls.setdefault(B,{});N=M.get(H,0);M[H]=N+1;O=C.total_tool_calls.setdefault(B,{});R=O.get(H,0);O[H]=R+1;C.curr_tool_trajectory.append(B)
				if N>0 and B not in A._allow_duplicate_tool_names:E.append(ToolMessage(content=format_tool_error(code='duplicate_tool_call',message='Skipped duplicate tool call with identical arguments.',details={_E:B,_H:G,_D:D[_B]}),tool_call_id=D[_B]));continue
				try:S=L.invoke(G)
				except Exception as P:E.append(ToolMessage(content=format_tool_error(code=type(P).__name__,message=str(P),details={_E:B,_D:D[_B],_H:G}),tool_call_id=D[_B]));continue
				A.process_tool_output(D,S,C,E);A.process_llm_output(D,C)
			for I in K:T=A.llm.parse_tool_args(I.get(_H));E.append(ToolMessage(content=format_tool_error(code='parallel_call_disallowed',message='Skipped additional tool call because only one tool is allowed per iteration. Resend the tool call in a new turn.',details={_E:I.get(_F),_D:I[_B],_H:T}),tool_call_id=I[_B]))
			for U in E:A._log_tool_error(U,J)
			C.messages.extend(E);return C
		def M(state:AgentState)->AgentState:A=state;B=HumanMessage(content='You must call a tool to proceed. If you have completed your task, call the finalize tool. Otherwise, call an appropriate tool to continue.');A.messages.append(B);return A
		def N(state:AgentState)->AgentState:B=state;RichLog.debug(f"[ReActAgent] force_end: triggering force finalize (iteration={B.iterations}, attempts={B.force_end_attempts})");return A._execute_force_end(B)
		def O(state:AgentState)->str:
			B=state;D=A._should_end_after_tools(B)
			if D:RichLog.debug(f"[ReActAgent] should_continue_after_tools: 'end' (_end_now=True, iteration={B.iterations})");return E
			if B.iterations>=A.max_iters:RichLog.debug(f"[ReActAgent] should_continue_after_tools: 'force_end' (iteration={B.iterations} >= max_iters={A.max_iters})");return C
			RichLog.debug(f"[ReActAgent] should_continue_after_tools: 'continue' (iteration={B.iterations})");return H
		def P(state:AgentState)->str:
			B=state;D:Optional[AIMessage]=_A
			for F in reversed(B.messages):
				if isinstance(F,AIMessage):D=F;break
			if D and D.tool_calls:G=[A.get(_F)for A in D.tool_calls];RichLog.debug(f"[ReActAgent] should_continue_after_llm: 'use_tools' (tools={G}, iteration={B.iterations})");return I
			if A.strict_finalize and not B.finalize_called:
				if B.no_tool_retries<A.max_no_tool_retries:B.no_tool_retries+=1;RichLog.debug(f"[ReActAgent] should_continue_after_llm: 'nudge' (no tool calls, retry {B.no_tool_retries}/{A.max_no_tool_retries}, iteration={B.iterations})");return J
				RichLog.debug(f"[ReActAgent] should_continue_after_llm: 'force_end' (strict_finalize=True, finalize_called=False, retries exhausted={B.no_tool_retries}, iteration={B.iterations})");return C
			if B.iterations>=A.max_iters:RichLog.debug(f"[ReActAgent] should_continue_after_llm: 'force_end' (iteration={B.iterations} >= max_iters={A.max_iters})");return C
			RichLog.debug(f"[ReActAgent] should_continue_after_llm: 'end' (finalize_called={B.finalize_called}, iteration={B.iterations})");return E
		B.add_node(D,K);B.add_node(F,L);B.add_node(G,M);B.add_node(C,N);B.set_entry_point(D);B.add_conditional_edges(D,P,{I:F,J:G,C:C,E:END});B.add_edge(G,D);B.add_edge(C,END);B.add_conditional_edges(F,O,{H:D,C:C,E:END});Q=MemorySaver()if A.use_checkpointer else _A;return B.compile(checkpointer=Q)
	def invoke(B,input_msg:str,state:Optional[AgentState]=_A,config:Optional[Dict[str,Any]]=_A)->AgentState:
		I=config;H=input_msg;G='recursion_limit';C=state;B._end_now=_G
		if C is _A:A=AgentState(messages=[SystemMessage(content=B.system_message),HumanMessage(content=H)],iterations=0)
		else:
			A=C.model_copy(deep=_C)if hasattr(C,'model_copy')else C
			if not A.messages or not isinstance(A.messages[0],SystemMessage):A.messages.insert(0,SystemMessage(content=B.system_message))
			A.messages.append(HumanMessage(content=H))
		E=5*B.max_iters
		if I is _A:D:Dict[str,Any]={'configurable':{'thread_id':'default'},G:E}
		else:
			D=dict(I);J=D.get(G)
			if J is _A or J<E:D[G]=E
		F=B.graph.invoke(A,config=D)
		if isinstance(F,dict):return AgentState(**F)
		return F