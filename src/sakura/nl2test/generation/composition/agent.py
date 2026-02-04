from __future__ import annotations
_P='test_code'
_O='source'
_N='compile_and_execute_test'
_M='view_test_code'
_L='composition_agent_gherkin_finalize.jinja2'
_K='message'
_J='comments'
_I='qualified_class_name'
_H='generate_test_code'
_G='name'
_F=False
_E='tool_call_id'
_D='tool'
_C=True
_B='id'
_A=None
import re
from pathlib import Path
from typing import Any,Dict,List,Optional,Tuple,Type
from langchain_core.messages import ToolMessage,ToolCall
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from sakura.nl2test.core.react_agent import ReActAgent
from sakura.nl2test.core.message_redactor import MessageRedactor
from sakura.nl2test.generation.common.cldk_normalizer import CLDKArgNormalizer
from sakura.nl2test.generation.common.compilation_execution import CompilationExecutionMixin
from sakura.nl2test.models import AgentState
from sakura.nl2test.models.agents import FinalizeCommentsArgs
from sakura.nl2test.prompts.load_prompt import LoadPrompt,PromptFormat
from sakura.utils.llm import LLMClient,FormatValidator
from sakura.utils.config import Config
from sakura.utils.constants import TEST_DIR
from sakura.utils.file_io.test_file_manager import TestFileManager,TestFileInfo
from sakura.utils.exceptions import ProjectCompilationError
from sakura.utils.tool_messages import format_tool_error,format_tool_ok
class CompositionReActAgent(ReActAgent,CompilationExecutionMixin):
	'\n    Composition agent that generates executable Java tests from localized scenarios.\n\n    Takes localized step sequences and generates compilable Java test code,\n    managing the test file lifecycle and validating compilation/execution.\n    ';_CODE_REDACTED='(redacted since new code generated)';_JAVA_PACKAGE_DECL_RE=re.compile('(?m)^\\s*package\\s+(?P<package>[A-Za-z_]\\w*(?:\\.[A-Za-z_]\\w*)*)\\s*;');_JAVA_TOP_LEVEL_TYPE_RE=re.compile('(?m)^(?:\\s*@\\w+(?:\\([^)]*\\))?\\s*)*(?:public\\s+)?(?:abstract\\s+|final\\s+)?(?:class|interface|enum|record)\\s+(?P<name>[A-Za-z_]\\w*)')
	@staticmethod
	def _extract_package_from_code(code:str)->str|_A:A=CompositionReActAgent._JAVA_PACKAGE_DECL_RE.search(code);return A.group('package')if A else _A
	@staticmethod
	def _extract_top_level_type_name(code:str)->str|_A:A=CompositionReActAgent._JAVA_TOP_LEVEL_TYPE_RE.search(code);return A.group(_G)if A else _A
	@staticmethod
	def _normalize_test_qualified_class_name(test_code:str,qcn:str)->str:
		C=test_code;B=qcn;F=CompositionReActAgent._extract_package_from_code(C)or'';D,G,A=B.rpartition('.')
		if not G:D='';A=B
		if not A:A=CompositionReActAgent._extract_top_level_type_name(C)or B
		E=D or F
		if E:return f"{E}.{A}"
		return A
	def __init__(B,*,llm:LLMClient,tools:List[BaseTool],allow_duplicate_tools:List[BaseTool]|_A=_A,system_message:str,project_root:Path,test_base_dir:str|Path|_A=_A,module_root:Path|_A=_A,max_iters:int=30,parallelizable:bool=_C):
		C=module_root;super().__init__(llm=llm,tools=tools,allow_duplicate_tools=allow_duplicate_tools,system_message=system_message,allow_parallelize=parallelizable,max_iters=max_iters);B.project_root=Path(project_root);B.test_base_dir=test_base_dir;A=Path(C).expanduser()if C is not _A else _A
		if A is not _A and not A.is_absolute():A=B.project_root/A
		B.module_root=A.resolve()if A is not _A else _A;B._code_iteration_counter=0
	def _get_finalize_schema(A)->Type[BaseModel]:return FinalizeCommentsArgs
	def _get_force_finalize_system_prompt(B)->str:A=LoadPrompt.load_prompt(_L,PromptFormat.JINJA2,'system');return A.format()
	def _get_force_finalize_chat_prompt(B)->str:A=LoadPrompt.load_prompt(_L,PromptFormat.JINJA2,'chat');return A.format()
	def _process_force_finalize_result(B,result:BaseModel,state:AgentState)->_A:A=state;A.finalize_called=_C;A.final_comments=getattr(result,_J,'')
	@staticmethod
	def _cleanup_empty_dirs(base_dir:Path,start_dir:Path)->_A:
		'Remove empty directories up to base_dir after file deletion.'
		try:B=base_dir.resolve();A=start_dir.resolve()
		except Exception:return
		while A!=B and B in A.parents:
			try:A.rmdir()
			except OSError:break
			A=A.parent
	def _get_test_file_manager(A)->TestFileManager:
		if A.project_root is _A:raise ValueError('project_root is not configured')
		B=A.test_base_dir or TEST_DIR;return TestFileManager(A.project_root,test_base_dir=B)
	def _save_code_iteration(A,test_code:str)->_A:
		'Save a code iteration to the code_iteration directory if enabled.'
		try:B=Config();D=B.get('composition','store_code_iteration')
		except Exception:return
		if not D:return
		try:E=Path(B.get('project','project_output_dir'))
		except Exception:return
		C=E/'code_iteration';C.mkdir(parents=_C,exist_ok=_C);F=f"nl2test_code_iteration_{A._code_iteration_counter}.java";(C/F).write_text(test_code,encoding='utf-8');A._code_iteration_counter+=1
	def prepare_tool_args(C,tool_name:str,raw_args:Dict[str,Any],state:AgentState)->Tuple[str,Dict[str,Any]]:'Normalize tool arguments for CLDK compatibility.';A=tool_name;D=state;B=CLDKArgNormalizer.normalize_args(A,raw_args);return A,B
	def process_tool_output(A,tool_call:ToolCall,result:Any,state:AgentState,outputs:List)->_A:
		'Process tool output and mutate state when appropriate.';D=result;C=outputs;B=tool_call
		if A._append_tool_error_if_needed(B,D,C):return
		E=B.get(_G);G={_H:A._process_generate_test_code_tool,_M:A._process_view_test_code_output,_N:A._process_compile_and_execute_test_output,'finalize':A._process_finalize_tool_output,'modify_scenario_comment':A._process_modify_scenario_comment_output};H=G.get(E,A._process_generic_tool_output)
		try:H(B,D,state,C)
		except ProjectCompilationError:raise
		except Exception as F:C.append(ToolMessage(content=format_tool_error(code=type(F).__name__,message=str(F),details={_D:E,_E:B[_B]}),tool_call_id=B[_B]))
	def process_llm_output(E,tool_call:ToolCall,state:AgentState)->_A:
		'Hook for processing LLM outputs (currently unused).';A=tool_call;C=A.get(_G);D={};B=D.get(C)
		if B:
			try:B(A,state)
			except Exception:pass
	def _redact_previous_test_code_results(A,state:AgentState)->_A:'Redact stale tool outputs whenever new test code is generated.';MessageRedactor.redact_tool_outputs_by_tool(state,tool_redactions={_M:{_O:A._CODE_REDACTED},_N:{'compilation':A._CODE_REDACTED,'execution':A._CODE_REDACTED}})
	def _redact_previous_test_code_inputs(A,state:AgentState)->_A:'Redact older generate_test_code inputs when new code is saved.';MessageRedactor.redact_tool_inputs(state,tool_name=_H,keys_to_redact={_P:A._CODE_REDACTED},llm_client=A.llm,exclude_latest=_C)
	def _process_generate_test_code_tool(D,tool_call:ToolCall,result:Dict[str,Any],state:AgentState,outputs:List)->_A:
		L=result;J=outputs;C=tool_call;A=state;B=L.get(_P)
		if isinstance(B,str):B=FormatValidator.sanitize_code_block(B)
		F=L.get(_I);G=L.get('method_signature')
		if not isinstance(B,str)or not B.strip():J.append(ToolMessage(content=format_tool_error(code='invalid_test_code',message='generate_test_code must provide non-empty Java source in test_code.',details={_D:_H,_E:C[_B]}),tool_call_id=C[_B]));return
		if not isinstance(F,str)or not F.strip():J.append(ToolMessage(content=format_tool_error(code='invalid_qualified_class_name',message='generate_test_code must provide a fully qualified test class name in qualified_class_name.',details={_D:_H,_E:C[_B]}),tool_call_id=C[_B]));return
		if not isinstance(G,str)or not G.strip():J.append(ToolMessage(content=format_tool_error(code='invalid_method_signature',message='generate_test_code must provide the generated @Test method signature in method_signature.',details={_D:_H,_E:C[_B]}),tool_call_id=C[_B]));return
		M=F.strip();N=D._normalize_test_qualified_class_name(B,M);F=N;G=G.strip();D._save_code_iteration(B);E=D._get_test_file_manager();K:Optional[TestFileInfo]=_A;H:Optional[Path]=_A
		if A.package and A.class_name:R=f"{A.package}.{A.class_name}"if A.package else A.class_name;K=TestFileInfo(qualified_class_name=R);H=E.target_path(K,encode_class_name=_F)
		O=TestFileInfo(qualified_class_name=F,test_code=B);S=E.target_path(O,encode_class_name=_F)
		if H is not _A and H==S:I,Q=E.save_single(O,encode_class_name=_F,allow_overwrite=_C,sync_names=_C)
		else:
			if K is not _A and H is not _A:E.delete_single(K,encode_class_name=_F,strict=_C,max_attempts=2,retry_delay=.05);D._cleanup_empty_dirs(E.test_base_dir,H.parent)
			I,Q=E.save_single(O,encode_class_name=_F,sync_names=_C)
		D._redact_previous_test_code_results(A);D._redact_previous_test_code_inputs(A);P:Dict[str,Any]={_K:'Saved test code.',_I:I,'path':str(Q)}
		if M!=N:P['input_qualified_class_name']=M;P['normalized_qualified_class_name']=N
		J.append(ToolMessage(content=format_tool_ok(P),tool_call_id=C[_B]))
		if'.'in I:T,U=I.rsplit('.',1);A.package=T or _A;A.class_name=U
		else:A.package=_A;A.class_name=I
		A.method_signature=G
	def _process_view_test_code_output(Q,tool_call:ToolCall,result:Dict[str,Any],state:AgentState,outputs:List)->_A:
		P='total_lines';O='invalid_line_range';L=result;K='end_line';H='start_line';C=outputs;B=state;A=tool_call;D=A[_G]
		if not B.class_name:C.append(ToolMessage(content=format_tool_error(code='no_active_test',message='No test code has been generated or saved.',details={_D:D,_E:A[_B]}),tool_call_id=A[_B]));return
		I=L.get(H);J=L.get(K)
		if not isinstance(I,int)or not isinstance(J,int):C.append(ToolMessage(content=format_tool_error(code=O,message='start_line and end_line must be integers.',details={_D:D,_E:A[_B],H:I,K:J}),tool_call_id=A[_B]));return
		E=I;R=J;F=f"{B.package}.{B.class_name}"if B.package else B.class_name;S=Q._get_test_file_manager();T=TestFileInfo(qualified_class_name=F)
		try:U=S.load(T,encode_class_name=_F)
		except FileNotFoundError:C.append(ToolMessage(content=format_tool_error(code='test_file_not_found',message=f"Test file not found for {F}.",details={_D:D,_E:A[_B],_I:F}),tool_call_id=A[_B]));return
		M=U.splitlines();G=len(M)
		if E>G:C.append(ToolMessage(content=format_tool_error(code=O,message='start_line is beyond the end of the file.',details={_D:D,_E:A[_B],H:E,P:G}),tool_call_id=A[_B]));return
		N=min(R,G);V='\n'.join(M[E-1:N]);W={_I:F,_O:V,P:G,H:E,K:N};C.append(ToolMessage(content=format_tool_ok(W),tool_call_id=A[_B]))
	def _process_compile_and_execute_test_output(A,tool_call:ToolCall,result:Any,state:AgentState,outputs:List)->_A:'Process compile_and_execute_test using shared mixin.';A.process_compile_and_execute(tool_call,state,outputs)
	def _process_finalize_tool_output(C,tool_call:ToolCall,result:Any,state:AgentState,outputs:List)->_A:B=result;A=state;D=B.get(_J,'')if isinstance(B,dict)else B;A.final_comments=str(D);A.finalize_called=_C;A.force_end_attempts=0;outputs.append(ToolMessage(content=format_tool_ok({_J:A.final_comments}),tool_call_id=tool_call[_B]));setattr(C,'_end_now',_C)
	def _process_modify_scenario_comment_output(R,tool_call:ToolCall,result:Any,state:AgentState,outputs:List)->_A:
		P='invalid_arguments';N='step_id';L='order';H=outputs;G=state;B=result;A=tool_call;J=A[_G];C:Optional[int]=_A;I:Optional[str]=_A;E:Optional[int]=_A;K:Optional[str]=_A
		if isinstance(B,(list,tuple))and len(B)==2:
			if G.atomic_blocks is not _A:E,K=int(B[0]),str(B[1])
			else:C,I=int(B[0]),str(B[1])
		elif isinstance(B,dict):C=B.get(_B);I=B.get('comment');E=B.get(L);K=B.get('note')
		if G.atomic_blocks is not _A:
			if E is _A or K is _A:H.append(ToolMessage(content=format_tool_error(code=P,message='Invalid modify_scenario_comment args for GRAMMATICAL mode.',details={_D:J,_E:A[_B],L:E}),tool_call_id=A[_B]));return
			D=_F
			for O in G.atomic_blocks.atomic_blocks:
				if O.order==int(E):O.notes=str(K);D=_C;break
			if D:H.append(ToolMessage(content=format_tool_ok({_K:f"Updated note for block order {E}.",L:E}),tool_call_id=A[_B]))
			else:H.append(ToolMessage(content=format_tool_error(code='block_not_found',message=f"No atomic block with order {E} exists.",details={_D:J,_E:A[_B],L:E}),tool_call_id=A[_B]))
			return
		if G.localized_scenario is not _A:
			if C is _A or I is _A:H.append(ToolMessage(content=format_tool_error(code=P,message='Invalid modify_scenario_comment args for GHERKIN mode.',details={_D:J,_E:A[_B],N:C}),tool_call_id=A[_B]));return
			D=_F
			for F in G.localized_scenario.setup:
				if F.id==int(C):F.comments=str(I);D=_C;break
			if not D:
				for M in G.localized_scenario.gherkin_groups:
					for Q in(M.given,M.when,M.then):
						for F in Q:
							if F.id==int(C):F.comments=str(I);D=_C;break
						if D:break
					if D:break
			if not D:
				for F in G.localized_scenario.teardown:
					if F.id==int(C):F.comments=str(I);D=_C;break
			if D:H.append(ToolMessage(content=format_tool_ok({_K:f"Updated comment for step id {C}.",N:C}),tool_call_id=A[_B]))
			else:H.append(ToolMessage(content=format_tool_error(code='step_not_found',message=f"No step with id {C} exists.",details={_D:J,_E:A[_B],N:C}),tool_call_id=A[_B]))
			return
		H.append(ToolMessage(content=format_tool_error(code='state_unavailable',message='No scenario or atomic blocks present to modify.',details={_D:J,_E:A[_B]}),tool_call_id=A[_B]))
	def _process_generic_tool_output(A,tool_call:ToolCall,result:Any,_:AgentState,outputs:List)->_A:A._append_tool_output(tool_call,result,outputs)