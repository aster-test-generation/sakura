_B='qualified_class_name'
_A='modifiers'
import textwrap
from typing import Any,Dict,List,Tuple,Union
from cldk.analysis.java import JavaAnalysis
from langchain_core.tools import BaseTool,StructuredTool
from sakura.nl2test.generation.common.tools.common_java_analysis import CommonJavaAnalysisTools
from sakura.nl2test.generation.common.tools.common_search import CommonSearchTools
from sakura.nl2test.generation.localization.tool_descriptions import CLASS_DETAILS_DESC,INHERITED_LIBRARY_CLASSES_DESC,QUERY_METHOD_DESC,SEARCH_REACHABLE_METHODS_DESC
from sakura.nl2test.models import QueryClassArgs,QueryVectorDataArgs,SearchReachableMethodsArgs
from sakura.nl2test.preprocessing.searchers import ClassSearcher,MethodSearcher
from sakura.utils.analysis import Reachability
from sakura.utils.exceptions import ClassNotFoundError,InvalidArgumentError,ToolExceptionHandler
class BaseLocalizationTools(CommonJavaAnalysisTools,CommonSearchTools):
	'Shared localization tools; subclasses implement finalize step.'
	def __init__(A,*,analysis:JavaAnalysis,method_searcher:MethodSearcher,class_searcher:ClassSearcher)->None:CommonJavaAnalysisTools.__init__(A,analysis=analysis);CommonSearchTools.__init__(A,class_searcher=class_searcher);A.method_searcher=method_searcher;(A.tools):List[BaseTool]=[A._make_query_method_tool(),A._make_query_class_tool(),A._make_search_reachable_methods_tool(),A._make_extract_code_tool(),A._make_get_method_details_tool(),A._make_get_inherited_library_classes_tool(),A._make_call_site_details_tool()];(A.allow_duplicate_tools):List[BaseTool]=[]
	def all(A)->Tuple[List[BaseTool],List[BaseTool]]:return A.tools,A.allow_duplicate_tools
	def _make_query_method_tool(A)->StructuredTool:
		def B(query:str,i:int,j:int)->List[Dict[str,str]]:
			if i<=0:raise InvalidArgumentError('i must be positive',extra_info={'i':i})
			if j<i:raise InvalidArgumentError('j must be greater than i',extra_info={'i':i,'j':j})
			return A.method_searcher.find_similar_in_range(query,i,j)
		return StructuredTool.from_function(func=B,name='query_method_db',description=textwrap.dedent(QUERY_METHOD_DESC).strip(),args_schema=QueryVectorDataArgs,handle_tool_error=ToolExceptionHandler.handle_error)
	def _make_search_reachable_methods_tool(J)->StructuredTool:
		def A(qualified_class_name:str,query:str,visibility_mode:str,k:int=5)->List[Dict[str,Any]]:
			P='requires_subclass';O='visibility';N='declaring_class_name';M='containing_class_name';I='method_signature';D=visibility_mode;C=qualified_class_name
			if D not in('public','same_package','same_package_or_subclass'):raise InvalidArgumentError('Invalid visibility mode',extra_info={'visibility_mode':D})
			if k<=0:raise InvalidArgumentError('k must be positive',extra_info={'k':k})
			Q=10;R=max(k*Q,k);S=J.method_searcher.find_similar(query,k=R);T=Reachability(J.analysis).get_visible_class_methods(C,visibility_mode=D,include_metadata=True);K:Dict[Tuple[str,str],Dict[str,Any]]={}
			for(U,V)in T.items():
				for A in V:
					if not isinstance(A,dict):continue
					B=A.get(I)
					if not B:continue
					K[U,B]=A
			E:List[Dict[str,Any]]=[];L:set[Tuple[str,str]]=set()
			for F in S:
				if F.get(M)!=C:continue
				G=F.get(N);B=F.get(I)
				if not G or not B:continue
				H=G,B
				if H in L:continue
				A=K.get(H)
				if not A:continue
				L.add(H);E.append({I:B,N:G,M:C,_A:A.get(_A,[]),O:A.get(O),P:A.get(P,False)})
				if len(E)>=k:break
			return E
		return StructuredTool.from_function(func=A,name='search_reachable_methods_in_class',description=textwrap.dedent(SEARCH_REACHABLE_METHODS_DESC).strip(),args_schema=SearchReachableMethodsArgs,handle_tool_error=ToolExceptionHandler.handle_error)
	def _make_get_class_details_tool(C)->StructuredTool:
		def A(qualified_class_name:str)->Dict[str,Union[str,List[str],None]]:
			B=qualified_class_name;A=C.analysis.get_class(B)
			if not A:raise ClassNotFoundError(f"Class {B} not found in application (may be externally defined or misspelled).",extra_info={_B:B})
			return{'class_name':B.split('.')[-1],_A:A.modifiers,'extends_list':A.extends_list,'implements_list':A.implements_list,'annotations':A.annotations}
		return StructuredTool.from_function(func=A,name='get_class_details',description=textwrap.dedent(CLASS_DETAILS_DESC).strip(),args_schema=QueryClassArgs,handle_tool_error=ToolExceptionHandler.handle_error)
	def _make_get_inherited_library_classes_tool(C)->StructuredTool:
		F=Reachability(C.analysis)
		def A(qualified_class_name:str)->List[str]:
			A=qualified_class_name;G=C.analysis.get_class(A)
			if not G:raise ClassNotFoundError(f"Class {A} not found in application (may be externally defined or misspelled).",extra_info={_B:A})
			H=F.get_inherited_classes_and_interfaces(A);D:set[str]=set();E:List[str]=[]
			for B in H:
				if B in D:continue
				D.add(B)
				if not C.analysis.get_class(B):E.append(B)
			return E
		return StructuredTool.from_function(func=A,name='get_inherited_library_classes',description=textwrap.dedent(INHERITED_LIBRARY_CLASSES_DESC).strip(),args_schema=QueryClassArgs,handle_tool_error=ToolExceptionHandler.handle_error)