from typing import Any,Dict
from langchain_core.tools import ToolException
from sakura.utils.tool_messages import build_tool_error
class ToolExceptionHandler:
	@staticmethod
	def handle_error(error:ToolException)->Dict[str,Any]:A=error;B=dict(getattr(A,'extra_info',{})or{});return build_tool_error(code=type(A).__name__,message=str(A),details=B)