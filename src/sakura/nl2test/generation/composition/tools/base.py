from __future__ import annotations
_D='method_signature'
_C='type'
_B=None
_A='qualified_class_name'
import textwrap
from pathlib import Path
from typing import Dict,List,Tuple,Union
from cldk.analysis.java import JavaAnalysis
from langchain_core.tools import BaseTool,StructuredTool
from sakura.nl2test.core.deferred_tool import DeferredTool
from sakura.nl2test.generation.common.tools.common_java_analysis import CommonJavaAnalysisTools
from sakura.nl2test.generation.common.tools.common_search import CommonSearchTools
from sakura.nl2test.generation.common.tools.common_test_tools import CommonTestTools
from sakura.nl2test.generation.composition.tool_descriptions import FINALIZE_DESC,GENERATE_TEST_CODE_DESC,GET_CLASS_CONSTRUCTORS_AND_FACTORIES_DESC,GET_CLASS_FIELDS_DESC,GET_CLASS_IMPORTS_DESC,GET_GETTERS_AND_SETTERS_DESC,GET_MAVEN_DEPENDENCIES_DESC
from sakura.nl2test.models import FinalizeCommentsArgs,GenerateTestCodeArgs,NL2TestInput,NoArgs,QueryClassArgs
from sakura.nl2test.preprocessing.searchers import ClassSearcher,MethodSearcher
from sakura.utils.analysis.java_analyzer import CommonAnalysis
from sakura.utils.exceptions import ClassNotFoundError,ToolExceptionHandler
from sakura.utils.file_io.pom_processor import PomProcessor
from sakura.utils.llm import LLMClient
class BaseCompositionTools(CommonJavaAnalysisTools,CommonSearchTools):
	'\n    Shared composition tools; subclasses can extend with mode-specific tools.\n\n    Provides tools for:\n    - Code analysis (class fields, imports, constructors, getters/setters)\n    - Test generation (generate_test_code - deferred to agent)\n    - Test validation (view_test_code, compile_and_execute_test - deferred to agent)\n    - Finalization (finalize - deferred to agent)\n    '
	def __init__(A,*,analysis:JavaAnalysis,method_searcher:MethodSearcher,class_searcher:ClassSearcher,structured_llm:LLMClient,project_root:str,module_root:str|Path|_B=_B,nl2_input:NL2TestInput)->_B:B=module_root;CommonJavaAnalysisTools.__init__(A,analysis=analysis);CommonSearchTools.__init__(A,class_searcher=class_searcher);A.structured_llm=structured_llm;A.method_searcher=method_searcher;(A.project_root):Path=Path(project_root);(A.module_root):Path=Path(B)if B is not _B else A.project_root;(A.nl2_input):NL2TestInput=nl2_input;(A.tools):List[BaseTool]=[A._make_extract_code_tool(),A._make_get_method_details_tool(),A._make_get_class_fields_tool(),A._make_get_class_constructors_and_factories_tool(),A._make_get_getters_and_setters_tool(),A._make_get_maven_dependencies_tool(),CommonTestTools.make_view_test_code_tool(),A._make_generate_test_tool(),CommonTestTools.make_compile_and_execute_test_tool(),A._make_finalize_tool(),A._make_call_site_details_tool()];(A.allow_duplicate_tools):List[BaseTool]=[CommonTestTools.make_view_test_code_tool(),CommonTestTools.make_compile_and_execute_test_tool()]
	def all(A)->Tuple[List[BaseTool],List[BaseTool]]:return A.tools,A.allow_duplicate_tools
	def _make_get_class_fields_tool(E)->StructuredTool:
		'Get field declarations for a class.'
		def A(qualified_class_name:str)->List[Dict[str,Union[str,List[str]]]]:
			A=qualified_class_name;C=E.analysis.get_class(A)
			if not C:raise ClassNotFoundError(f"Class {A} not found in application (may be externally defined or misspelled).",extra_info={_A:A})
			D=[]
			for B in C.field_declarations:D.append({'variable_names':B.variables,_C:B.type,'modifiers':B.modifiers})
			return D
		return StructuredTool.from_function(func=A,name='get_class_fields',description=textwrap.dedent(GET_CLASS_FIELDS_DESC).strip(),args_schema=QueryClassArgs,handle_tool_error=ToolExceptionHandler.handle_error)
	def _make_get_class_imports_tool(B)->StructuredTool:
		'Get imports for a class.'
		def A(qualified_class_name:str)->List[str]:
			A=qualified_class_name;C=B.analysis.get_class(A)
			if not C:raise ClassNotFoundError(f"Class {A} not found in application (may be externally defined or misspelled).",extra_info={_A:A})
			return CommonAnalysis(B.analysis).get_imports_for_class(A)
		return StructuredTool.from_function(func=A,name='get_class_imports',description=textwrap.dedent(GET_CLASS_IMPORTS_DESC).strip(),args_schema=QueryClassArgs,handle_tool_error=ToolExceptionHandler.handle_error)
	def _make_get_class_constructors_and_factories_tool(C)->StructuredTool:
		'Get constructors and factory methods for a class.'
		def A(qualified_class_name:str)->List[Dict[str,str]]:
			A=qualified_class_name;G=C.analysis.get_class(A)
			if not G:raise ClassNotFoundError(f"Class {A} not found in application (may be externally defined or misspelled).",extra_info={_A:A})
			H:List[str]=list(C.analysis.get_constructors(A));F:List[str]=[]
			for B in C.analysis.get_methods_in_class(A):
				D=C.analysis.get_method(A,B)
				if not D:continue
				if'static'in D.modifiers and A==D.return_type:F.append(B)
			E:List[Dict[str,str]]=[]
			for B in H:E.append({_D:B,_C:'constructor'})
			for B in F:E.append({_D:B,_C:'factory'})
			return E
		return StructuredTool.from_function(func=A,name='get_class_constructors_and_factories',description=textwrap.dedent(GET_CLASS_CONSTRUCTORS_AND_FACTORIES_DESC).strip(),args_schema=QueryClassArgs,handle_tool_error=ToolExceptionHandler.handle_error)
	def _make_get_getters_and_setters_tool(B)->StructuredTool:
		'Get getter and setter methods for a class.'
		def A(qualified_class_name:str)->List[str]:
			A=qualified_class_name;F=B.analysis.get_class(A)
			if not F:raise ClassNotFoundError(f"Class {A} not found in application (may be externally defined or misspelled).",extra_info={_A:A})
			C:List[str]=[]
			for D in B.analysis.get_methods_in_class(A):
				E=B.analysis.get_method(A,D)
				if not E:continue
				if CommonAnalysis.is_getter_or_setter(E):C.append(D)
			return C
		return StructuredTool.from_function(func=A,name='get_getters_and_setters',description=textwrap.dedent(GET_GETTERS_AND_SETTERS_DESC).strip(),args_schema=QueryClassArgs,handle_tool_error=ToolExceptionHandler.handle_error)
	def _make_get_maven_dependencies_tool(A)->StructuredTool:
		'Get Maven dependencies from pom.xml.'
		def B()->List[Dict[str,str]]:B=PomProcessor.identify_dependencies(module_root=A.module_root,parent_roots=[A.project_root]);return[{'group_id':A.group_id,'artifact_id':A.artifact_id}for A in B]
		return StructuredTool.from_function(func=B,name='get_maven_dependencies',description=textwrap.dedent(GET_MAVEN_DEPENDENCIES_DESC).strip(),args_schema=NoArgs,handle_tool_error=ToolExceptionHandler.handle_error)
	def _make_generate_test_tool(A)->BaseTool:"\n        Create the generate_test_code tool.\n\n        This is a deferred tool - it returns all inputs and the agent's\n        process_tool_output hook handles file saving and state updates.\n        ";return DeferredTool.create(name='generate_test_code',description=textwrap.dedent(GENERATE_TEST_CODE_DESC).strip(),args_schema=GenerateTestCodeArgs,returns_input_keys=['test_code',_A,_D],processing_note='Agent saves test file to filesystem and updates state.package, state.class_name, state.method_signature')
	def _make_finalize_tool(A)->BaseTool:"\n        Create the finalize tool.\n\n        This is a deferred tool - it returns comments and the agent's\n        process_tool_output hook sets final state and ends the run.\n        ";return DeferredTool.create(name='finalize',description=textwrap.dedent(FINALIZE_DESC).strip(),args_schema=FinalizeCommentsArgs,returns_input_keys=['comments'],processing_note='Agent sets state.final_comments, state.finalize_called=True, and signals end of execution')