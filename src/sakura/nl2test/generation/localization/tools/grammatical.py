import textwrap
from typing import Tuple
from cldk.analysis.java import JavaAnalysis
from sakura.nl2test.preprocessing.searchers import MethodSearcher,ClassSearcher
from langchain_core.tools import StructuredTool
from sakura.nl2test.generation.localization.tool_descriptions import FINALIZE_ATOMIC_BLOCKS_DESC
from sakura.nl2test.models import AtomicBlockList,FinalizeAtomicBlockArgs
from sakura.utils.exceptions import ToolExceptionHandler
from sakura.utils.exceptions.tool_exceptions import BlockNotFoundError
from.base import BaseLocalizationTools
class GrammaticalLocalizationTools(BaseLocalizationTools):
	def __init__(A,*,analysis:JavaAnalysis,method_searcher:MethodSearcher,class_searcher:ClassSearcher)->None:super().__init__(analysis=analysis,method_searcher=method_searcher,class_searcher=class_searcher);A.tools.append(A._make_finalize_tool())
	def _make_finalize_tool(B)->StructuredTool:
		def A(current_blocks:AtomicBlockList,comments:str)->Tuple[AtomicBlockList,str]:
			A=current_blocks
			if A is None:raise BlockNotFoundError('Current blocks not found',extra_info={'current_blocks':A})
			return A,comments
		return StructuredTool.from_function(func=A,name='finalize',description=textwrap.dedent(FINALIZE_ATOMIC_BLOCKS_DESC).strip(),args_schema=FinalizeAtomicBlockArgs,handle_tool_error=ToolExceptionHandler.handle_error)