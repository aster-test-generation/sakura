from __future__ import annotations
_E='parameter_types'
_D='return_type'
_C='modifiers'
_B='method_signature'
_A='qualified_class_name'
import textwrap
from typing import Any,Dict,List,Union
from cldk.models.java.models import JMethodDetail,JCallable
from langchain_core.tools import StructuredTool
from sakura.utils.analysis.java_analyzer import CommonAnalysis
from sakura.utils.exceptions import ToolExceptionHandler,MethodNotFoundError,ClassNotFoundError
from sakura.nl2test.generation.common.tool_descriptions import EXTRACT_CODE_DESC,METHOD_DETAILS_DESC,CALL_SITE_DETAILS_DESC
from sakura.nl2test.models.agents import QueryMethodArgs,ExtractMethodCodeArgs
from cldk.analysis.java import JavaAnalysis
class CommonJavaAnalysisTools:
	'\n    Shared Java static-analysis tools used by both\n    localization and composition tool builders.\n    '
	def __init__(A,*,analysis:JavaAnalysis)->None:A.analysis=analysis
	def _make_extract_code_tool(G)->StructuredTool:
		def A(qualified_class_name:str,method_signature:str,start_line:int,end_line:int)->Dict[str,Any]:
			E=method_signature;A=qualified_class_name;J=G.analysis.get_class(A)
			if not J:raise ClassNotFoundError(f"Class {A} not found in application (may be externally defined or misspelled).",extra_info={_A:A})
			F=G.analysis.get_method(A,E)
			if not F:raise MethodNotFoundError(f"Method {E} not found in application class {A} (may be externally defined or misspelled).",extra_info={_A:A,_B:E})
			K=CommonAnalysis.get_complete_method_code(F.declaration,F.code);H=K.splitlines();D=len(H);B=max(1,start_line);C=max(0,end_line)
			if D==0:B=1;C=0
			else:B=min(B,D);C=min(C,D)
			I=B>C;L:List[str]=[]if I else H[B-1:C];M=''if not I else'start_line greater than end_line; returning empty slice.';return{'source':'\n'.join(L),'start_line':B,'end_line':C,'total_lines':D,'note':M}
		return StructuredTool.from_function(func=A,name='extract_method_code',description=textwrap.dedent(EXTRACT_CODE_DESC).strip(),args_schema=ExtractMethodCodeArgs,handle_tool_error=ToolExceptionHandler.handle_error)
	def _make_get_method_details_tool(D)->StructuredTool:
		def A(qualified_class_name:str,method_signature:str)->Dict[str,any]:
			C=method_signature;A=qualified_class_name;E=D.analysis.get_class(A)
			if not E:raise ClassNotFoundError(f"Class {A} not found in application (may be externally defined or misspelled).",extra_info={_A:A})
			B=D.analysis.get_method(A,C)
			if not B:raise MethodNotFoundError(f"Method {C} not found in application class {A} (may be externally defined or misspelled).",extra_info={_A:A,_B:C})
			F=CommonAnalysis(D.analysis);G=F.get_method_visibility(A,C);return{_B:B.signature,_C:B.modifiers,_D:B.return_type,_E:[A.type for A in B.parameters],'comments':[A.content[:25]for A in B.comments if A.content],'visibility':G}
		return StructuredTool.from_function(func=A,name='get_method_details',description=textwrap.dedent(METHOD_DETAILS_DESC).strip(),args_schema=QueryMethodArgs,handle_tool_error=ToolExceptionHandler.handle_error)
	def _make_call_site_details_tool(D)->StructuredTool:
		def A(qualified_class_name:str,method_signature:str)->List[Dict[str,Union[str,List[str]]]]:
			C=method_signature;A=qualified_class_name;H=D.analysis.get_class(A)
			if not H:raise ClassNotFoundError(f"Class {A} not found in application (may be externally defined or misspelled).",extra_info={_A:A})
			B=D.analysis.get_method(A,C)
			if not B:raise MethodNotFoundError(f"Call sites could not be found because method {C} not found in application class {A} (may be externally defined or misspelled).",extra_info={_A:A,_B:C})
			I=D.analysis.get_callees(source_class_name=A,source_method_declaration=C,using_symbol_table=True).get('callee_details',[]);E:List[Dict[str,Any]]=[]
			for F in I:G:JMethodDetail=F['callee_method'];B:JCallable=G.method;J=F.get('calling_lines',[]);K=max(len(J),1);E.append({_A:G.klass,_B:B.signature,_D:B.return_type,_E:[A.type for A in B.parameters],_C:B.modifiers,'num_times_called':K})
			return E
		return StructuredTool.from_function(func=A,name='get_call_site_details',description=textwrap.dedent(CALL_SITE_DETAILS_DESC).strip(),args_schema=QueryMethodArgs,handle_tool_error=ToolExceptionHandler.handle_error)