from __future__ import annotations
import textwrap
from langchain_core.tools import StructuredTool
from sakura.nl2test.models import ViewTestCodeArgs,NoArgs
from sakura.nl2test.core.deferred_tool import DeferredTool
from sakura.nl2test.generation.common.tool_descriptions import VIEW_TEST_CODE_DESC,COMPILE_AND_EXECUTE_TEST_DESC
from sakura.utils.exceptions import ToolExceptionHandler
class CommonTestTools:
	"\n    Shared test-related tools used by both supervisor and composition agents.\n\n    These are deferred tools - they return minimal data and actual processing\n    is done in the agent's process_tool_output hook.\n    "
	@staticmethod
	def make_view_test_code_tool()->StructuredTool:
		"\n        Create the view_test_code tool.\n\n        This is a deferred tool - it validates inputs and returns them.\n        Actual file loading is done in the agent's process_tool_output hook.\n        "
		def A(start_line:int,end_line:int)->dict:
			B=end_line;A=start_line
			if A<1 or B<1:raise ValueError('start_line and end_line must be >= 1.')
			if B<A:raise ValueError('end_line must be >= start_line.')
			return{'start_line':A,'end_line':B}
		A.__doc__='DEFERRED TOOL: Returns line range for agent-side file loading.\nAgent hook loads test file and slices by line numbers.';return StructuredTool.from_function(func=A,name='view_test_code',description=textwrap.dedent(VIEW_TEST_CODE_DESC).strip(),args_schema=ViewTestCodeArgs,handle_tool_error=ToolExceptionHandler.handle_error)
	@staticmethod
	def make_compile_and_execute_test_tool()->StructuredTool:'\n        Create the compile_and_execute_test tool.\n\n        This is a deferred tool with no arguments - returns empty dict.\n        Actual Maven compilation and test execution is done in agent hook.\n        ';return DeferredTool.create_no_args(name='compile_and_execute_test',description=textwrap.dedent(COMPILE_AND_EXECUTE_TEST_DESC).strip(),returns_static={},processing_note='Agent runs Maven compilation and test execution using state.class_name and state.method_signature')