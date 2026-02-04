from __future__ import annotations
_D='comments'
_C='localization_agent_gherkin_finalize.jinja2'
_B=True
_A=None
from typing import Any,Dict,List,Tuple,Type
from langchain_core.messages import ToolMessage,ToolCall
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from sakura.nl2test.core.react_agent import ReActAgent
from sakura.nl2test.generation.common.cldk_normalizer import CLDKArgNormalizer
from sakura.nl2test.models import AgentState
from sakura.nl2test.models.agents import FinalizeScenarioArgs,FinalizeAtomicBlockArgs
from sakura.nl2test.models.decomposition import DecompositionMode
from sakura.nl2test.prompts.load_prompt import LoadPrompt,PromptFormat
from sakura.utils.llm import LLMClient
from sakura.utils.tool_messages import format_tool_ok,format_tool_error
class LocalizationReActAgent(ReActAgent):
	'\n    Localization agent that maps natural language steps to relevant code methods.\n\n    Decomposes and localizes test description steps to concrete methods in the\n    target Java application using vector search and static analysis.\n    '
	def __init__(A,*,llm:LLMClient,tools:List[BaseTool],allow_duplicate_tools:List[BaseTool]|_A=_A,system_message:str,max_iters:int=30,decomposition_mode:DecompositionMode=DecompositionMode.GRAMMATICAL,parallelizable:bool=_B):super().__init__(llm=llm,tools=tools,allow_duplicate_tools=allow_duplicate_tools,system_message=system_message,allow_parallelize=parallelizable,max_iters=max_iters);A.decomposition_mode=decomposition_mode
	def _get_finalize_schema(A)->Type[BaseModel]:
		if A.decomposition_mode==DecompositionMode.GHERKIN:return FinalizeScenarioArgs
		return FinalizeAtomicBlockArgs
	def _get_force_finalize_system_prompt(A)->str:
		if A.decomposition_mode==DecompositionMode.GHERKIN:B=LoadPrompt.load_prompt(_C,PromptFormat.JINJA2,'system');return B.format()
		return'You must produce the finalize output for the localization task. Use the message history to extract the current state of the localized steps.'
	def _get_force_finalize_chat_prompt(A)->str:
		if A.decomposition_mode==DecompositionMode.GHERKIN:B=LoadPrompt.load_prompt(_C,PromptFormat.JINJA2,'chat');return B.format()
		return'Produce the final localized blocks based on the conversation history.'
	def _process_force_finalize_result(G,result:BaseModel,state:AgentState)->_A:
		F='current_blocks';E='localized_scenario';B=state;A=result;B.finalize_called=_B;B.final_comments=getattr(A,_D,'')
		if G.decomposition_mode==DecompositionMode.GHERKIN:
			if hasattr(A,E):C=getattr(A,E);C.enforce_candidate_limits();B.localized_scenario=C
		elif hasattr(A,F):D=getattr(A,F);D.enforce_candidate_limits();B.atomic_blocks=D
	def prepare_tool_args(C,tool_name:str,raw_args:Dict[str,Any],state:AgentState)->Tuple[str,Dict[str,Any]]:'Normalize tool arguments for CLDK compatibility.';A=tool_name;D=state;B=CLDKArgNormalizer.normalize_args(A,raw_args);return A,B
	def process_tool_output(B,tool_call:ToolCall,result:Any,state:AgentState,outputs:List)->_A:
		D=result;C=outputs;A=tool_call
		if B._append_tool_error_if_needed(A,D,C):return
		E=A['name'];G={'finalize':B._process_finalize_tool_output}.get(E,B._process_generic_tool_output)
		try:G(A,D,state,C)
		except Exception as F:C.append(ToolMessage(content=format_tool_error(code=type(F).__name__,message=str(F),details={'tool':E,'tool_call_id':A['id']}),tool_call_id=A['id']))
	def _process_finalize_tool_output(C,tool_call:ToolCall,result:Any,state:AgentState,outputs:List)->_A:
		A=state;B,D=result;A.final_comments=str(D);A.finalize_called=_B;A.force_end_attempts=0
		if C.decomposition_mode==DecompositionMode.GHERKIN:B.enforce_candidate_limits();A.localized_scenario=B
		else:B.enforce_candidate_limits();A.atomic_blocks=B
		outputs.append(ToolMessage(content=format_tool_ok({_D:str(D)}),tool_call_id=tool_call['id']));setattr(C,'_end_now',_B)
	def _process_generic_tool_output(A,tool_call:ToolCall,result:Any,_:AgentState,outputs:List)->_A:A._append_tool_output(tool_call,result,outputs)