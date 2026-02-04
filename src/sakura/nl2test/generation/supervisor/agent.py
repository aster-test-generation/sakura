from __future__ import annotations
_T='missing_blocks'
_S='instructions'
_R='method_signature'
_Q='class_name'
_P='package'
_O='source'
_N='compile_and_execute_test'
_M='view_test_code'
_L='call_localization_agent'
_K='Supervisor uses _execute_force_end override; prompt methods are unused.'
_J='comments'
_I='scenario'
_H='blocks'
_G='call_composition_agent'
_F='name'
_E=True
_D='tool_call_id'
_C='tool'
_B='id'
_A=None
from typing import List,Any,Dict,Tuple,Optional,Type
from pathlib import Path
from langchain_core.messages import ToolMessage,ToolCall
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from sakura.nl2test.core.react_agent import ReActAgent
from sakura.nl2test.core.message_redactor import MessageRedactor
from sakura.nl2test.generation.common.compilation_execution import CompilationExecutionMixin
from sakura.nl2test.generation.composition.orchestrators.base import BaseCompositionOrchestrator
from sakura.nl2test.generation.localization.orchestrators.base import BaseLocalizationOrchestrator
from sakura.nl2test.models import AgentState
from sakura.nl2test.models.agents import FinalizeCommentsArgs
from sakura.utils.constants import TEST_DIR
from sakura.utils.llm import LLMClient
from sakura.utils.file_io.test_file_manager import TestFileManager,TestFileInfo
from sakura.utils.exceptions import ProjectCompilationError
from sakura.utils.tool_messages import format_tool_error,format_tool_ok
class SupervisorReActAgent(ReActAgent,CompilationExecutionMixin):
	'\n    Supervisor agent that orchestrates localization and composition sub-agents.\n\n    Coordinates the end-to-end conversion of natural language test descriptions\n    into executable Java tests by delegating to specialized sub-agents.\n    ';_BLOCK_REDACTED='(redacted due to updated blocks)';_SCENARIO_REDACTED='(redacted due to updated scenario)';_CODE_REDACTED='(redacted since new code generated)'
	def __init__(A,*,llm:LLMClient,tools:List[BaseTool],system_message:str,nl_description:str,project_root:Path|str|_A=_A,test_base_dir:str|Path|_A=_A,module_root:Path|_A=_A,allow_duplicate_tools:List[BaseTool]|_A=_A,max_iters:int=10,localization_agent:BaseLocalizationOrchestrator|_A=_A,composition_agent:BaseCompositionOrchestrator|_A=_A,parallelizable:bool=_E,**E)->_A:
		D=module_root;C=project_root;super().__init__(llm=llm,tools=tools,allow_duplicate_tools=allow_duplicate_tools,system_message=system_message,allow_parallelize=parallelizable,max_iters=max_iters,**E);A._nl_description=nl_description;A.project_root=Path(C)if C is not _A else _A;A.test_base_dir=test_base_dir;B=Path(D).expanduser()if D is not _A else _A
		if B is not _A and not B.is_absolute()and A.project_root is not _A:B=A.project_root/B
		A.module_root=B.resolve()if B is not _A else _A;A.localization_agent=localization_agent;A.composition_agent=composition_agent;(A.localization_state):Optional[AgentState]=_A;(A.composition_state):Optional[AgentState]=_A
	def _get_finalize_schema(A)->Type[BaseModel]:return FinalizeCommentsArgs
	def _execute_force_end(B,state:AgentState)->AgentState:'Skip LLM call for supervisor and finalize with concise comments.';A=state;A.force_end_attempts+=1;A.finalize_called=_E;A.final_comments='No issues.';B._log_force_finalize(A);return A
	def _get_force_finalize_system_prompt(A)->str:raise NotImplementedError(_K)
	def _get_force_finalize_chat_prompt(A)->str:raise NotImplementedError(_K)
	def _process_force_finalize_result(A,result:BaseModel,state:AgentState)->_A:raise NotImplementedError('Supervisor uses _execute_force_end override; this method is unused.')
	def prepare_tool_args(A,tool_name:str,raw_args:Dict[str,Any],state:AgentState)->Tuple[str,Dict[str,Any]]:return tool_name,raw_args
	def _clean_agent(C,state:Optional[AgentState],orchestrator:BaseLocalizationOrchestrator|BaseCompositionOrchestrator)->Optional[AgentState]:
		'Reset sub-agent state for fresh invocation.';A=state;orchestrator.reset_agent()
		if A is _A:return
		B=A.model_copy(deep=_E);B.reset_message_history();return B
	def _get_test_file_manager(A)->TestFileManager:B=A.test_base_dir or TEST_DIR;C=A.project_root or Path('.');return TestFileManager(C,test_base_dir=B)
	def process_tool_output(A,tool_call:ToolCall,result:Any,state:AgentState,outputs:List)->_A:
		D=result;C=outputs;B=tool_call
		if A._append_tool_error_if_needed(B,D,C):return
		E=B[_F];G={_L:A._process_call_localization_agent_output,_G:A._process_call_composition_agent_output,_M:A._process_view_test_code_output,_N:A._process_compile_and_execute_test_output,'finalize':A._process_finalize_tool_output};H=G.get(E,A.process_generic_tool_output)
		try:H(B,D,state,C)
		except ProjectCompilationError:raise
		except Exception as F:C.append(ToolMessage(content=format_tool_error(code=type(F).__name__,message=str(F),details={_C:E,_D:B[_B]}),tool_call_id=B[_B]))
	def _redact_previous_block_outputs(A,state:AgentState)->_A:'Redact stale block/scenario payloads when new results are saved.';MessageRedactor.redact_tool_outputs(state,tool_names={_L,_G},keys_to_redact={_H:A._BLOCK_REDACTED,_I:A._SCENARIO_REDACTED})
	def _redact_previous_test_code_results(A,state:AgentState)->_A:'Redact outdated tool outputs when new composition results arrive.';MessageRedactor.redact_tool_outputs_by_tool(state,tool_redactions={_M:{_O:A._CODE_REDACTED},_N:{'compilation':A._CODE_REDACTED,'execution':A._CODE_REDACTED},_G:{_P:A._CODE_REDACTED,_Q:A._CODE_REDACTED,_R:A._CODE_REDACTED}})
	def _process_call_localization_agent_output(C,tool_call:ToolCall,result:Dict[str,Any],state:AgentState,outputs:List)->_A:
		J='enforce_candidate_limits';E=outputs;D=state;B=tool_call;F=B[_F];K=result.get(_S)or'';G=C.localization_agent
		if G is _A:E.append(ToolMessage(content=format_tool_error(code='localization_agent_not_configured',message='Localization agent is not configured on the supervisor.',details={_C:F,_D:B[_B]}),tool_call_id=B[_B]));return
		L=C._clean_agent(C.localization_state,G);I=D.localized_scenario or D.atomic_blocks
		if I is _A:E.append(ToolMessage(content=format_tool_error(code=_T,message='Supervisor has no blocks to inject into call_localization_agent. Ensure the supervisor state is initialized.',details={_C:F,_D:B[_B]}),tool_call_id=B[_B]));return
		try:A:AgentState=G.assign_task(I,instructions=K,agent_state=L)
		except Exception as M:E.append(ToolMessage(content=format_tool_error(code='localization_agent_failed',message=str(M),details={_C:F,_D:B[_B]}),tool_call_id=B[_B]));return
		if A.curr_tool_trajectory:A.tool_trajectories.append(A.curr_tool_trajectory.copy());A.curr_tool_trajectory.clear()
		C.localization_state=A
		if A.atomic_blocks is not _A:
			if hasattr(A.atomic_blocks,J):A.atomic_blocks.enforce_candidate_limits()
			D.atomic_blocks=A.atomic_blocks
		if A.localized_scenario is not _A:
			if hasattr(A.localized_scenario,J):A.localized_scenario.enforce_candidate_limits()
			D.localized_scenario=A.localized_scenario
		H:Dict[str,Any]={_J:str(A.final_comments or'')}
		if A.atomic_blocks is not _A:H[_H]=A.atomic_blocks.model_dump()
		elif A.localized_scenario is not _A:H[_I]=A.localized_scenario.model_dump()
		C._redact_previous_block_outputs(D);E.append(ToolMessage(content=format_tool_ok(H),tool_call_id=B[_B]))
	def _process_call_composition_agent_output(D,tool_call:ToolCall,result:Dict[str,Any],state:AgentState,outputs:List)->_A:
		E=outputs;C=tool_call;B=state;F=C[_F];J=result.get(_S)or'';G=D.composition_agent
		if G is _A:E.append(ToolMessage(content=format_tool_error(code='composition_agent_not_configured',message='Composition agent is not configured on the supervisor.',details={_C:F,_D:C[_B]}),tool_call_id=C[_B]));return
		K=D._clean_agent(D.composition_state,G);I=B.localized_scenario or B.atomic_blocks
		if I is _A:E.append(ToolMessage(content=format_tool_error(code=_T,message='Supervisor has no scenario to inject into call_composition_agent. Ensure the supervisor state is initialized.',details={_C:F,_D:C[_B]}),tool_call_id=C[_B]));return
		try:A:AgentState=G.assign_task(I,instructions=J,agent_state=K)
		except Exception as L:E.append(ToolMessage(content=format_tool_error(code='composition_agent_failed',message=str(L),details={_C:F,_D:C[_B]}),tool_call_id=C[_B]));return
		if A.curr_tool_trajectory:A.tool_trajectories.append(A.curr_tool_trajectory.copy());A.curr_tool_trajectory.clear()
		D.composition_state=A
		if A.atomic_blocks is not _A:A.atomic_blocks.enforce_candidate_limits();B.atomic_blocks=A.atomic_blocks
		if A.localized_scenario is not _A:A.localized_scenario.enforce_candidate_limits();B.localized_scenario=A.localized_scenario
		B.package=A.package;B.class_name=A.class_name;B.method_signature=A.method_signature;H:Dict[str,Any]={_J:str(A.final_comments or''),_P:A.package,_Q:A.class_name,_R:A.method_signature}
		if A.atomic_blocks is not _A:H[_H]=A.atomic_blocks.model_dump()
		elif A.localized_scenario is not _A:H[_I]=A.localized_scenario.model_dump()
		D._redact_previous_block_outputs(B);D._redact_previous_test_code_results(B);E.append(ToolMessage(content=format_tool_ok(H),tool_call_id=C[_B]))
	def _process_view_test_code_output(Q,tool_call:ToolCall,result:Dict[str,Any],state:AgentState,outputs:List)->_A:
		P='total_lines';O='qualified_class_name';N='invalid_line_range';K=result;J='end_line';H='start_line';D=outputs;C=state;A=tool_call;E=A[_F]
		if not C.class_name:D.append(ToolMessage(content=format_tool_error(code='no_active_test',message='No test code has been generated or saved.',details={_C:E,_D:A[_B]}),tool_call_id=A[_B]));return
		B=K.get(H);I=K.get(J)
		if not isinstance(B,int)or not isinstance(I,int):D.append(ToolMessage(content=format_tool_error(code=N,message='start_line and end_line must be integers.',details={_C:E,_D:A[_B],H:B,J:I}),tool_call_id=A[_B]));return
		F=f"{C.package}.{C.class_name}"if C.package else C.class_name;R=Q._get_test_file_manager();S=TestFileInfo(qualified_class_name=F)
		try:T=R.load(S,encode_class_name=False)
		except FileNotFoundError:D.append(ToolMessage(content=format_tool_error(code='test_file_not_found',message=f"Test file not found for {F}.",details={_C:E,_D:A[_B],O:F}),tool_call_id=A[_B]));return
		L=T.splitlines();G=len(L)
		if B>G:D.append(ToolMessage(content=format_tool_error(code=N,message='start_line is beyond the end of the file.',details={_C:E,_D:A[_B],H:B,P:G}),tool_call_id=A[_B]));return
		M=min(I,G);U='\n'.join(L[B-1:M]);V={O:F,_O:U,P:G,H:B,J:M};D.append(ToolMessage(content=format_tool_ok(V),tool_call_id=A[_B]))
	def _process_compile_and_execute_test_output(A,tool_call:ToolCall,result:Any,state:AgentState,outputs:List)->_A:'Process compile_and_execute_test using shared mixin.';A.process_compile_and_execute(tool_call,state,outputs)
	def _process_finalize_tool_output(C,tool_call:ToolCall,result:Any,state:AgentState,outputs:List)->_A:B=result;A=state;A.final_comments=str(B)if B is not _A else A.final_comments or'';A.finalize_called=_E;A.force_end_attempts=0;outputs.append(ToolMessage(content=format_tool_ok({_J:str(A.final_comments or'finalized')}),tool_call_id=tool_call[_B]));setattr(C,'_end_now',_E)
	def process_generic_tool_output(A,tool_call:ToolCall,result:Any,_:AgentState,outputs:List)->_A:A._append_tool_output(tool_call,result,outputs)