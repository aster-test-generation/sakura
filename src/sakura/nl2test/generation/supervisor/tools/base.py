from __future__ import annotations
from pathlib import Path
import textwrap
from typing import List,Tuple,Union
from langchain_core.tools import BaseTool
from sakura.nl2test.core.deferred_tool import DeferredTool
from sakura.nl2test.models import FinalizeCommentsArgs
from sakura.utils.llm import LLMClient
from sakura.nl2test.generation.supervisor.tool_descriptions import FINALIZE_DESC
from sakura.nl2test.generation.common.tools.common_test_tools import CommonTestTools
class BaseSupervisorTools:
	'\n    Base tool builder for supervisor agent.\n\n    Provides common tools shared across decomposition modes:\n    - view_test_code: View generated test source (deferred to agent)\n    - compile_and_execute_test: Compile and run tests (deferred to agent)\n    - finalize: End supervision\n\n    Subclasses add mode-specific delegation tools.\n    '
	def __init__(A,*,llm:LLMClient,project_root:Union[str,Path])->None:A.llm=llm;A.project_root=Path(project_root);(A.tools):List[BaseTool]=[CommonTestTools.make_view_test_code_tool(),CommonTestTools.make_compile_and_execute_test_tool(),A._make_finalize_tool()];(A.allow_duplicate_tools):List[BaseTool]=[CommonTestTools.make_view_test_code_tool(),CommonTestTools.make_compile_and_execute_test_tool()]
	def all(A)->Tuple[List[BaseTool],List[BaseTool]]:return A.tools,A.allow_duplicate_tools
	def _make_finalize_tool(A)->BaseTool:'Create the finalize tool to end supervision.';return DeferredTool.create(name='finalize',description=textwrap.dedent(FINALIZE_DESC).strip(),args_schema=FinalizeCommentsArgs,returns_input_keys=['comments'],processing_note='Agent sets finalize_called=True and ends the run')