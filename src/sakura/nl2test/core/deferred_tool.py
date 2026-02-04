from __future__ import annotations
_A=None
from typing import Any,Callable,Dict,List,Optional,Type
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from sakura.utils.exceptions import ToolExceptionHandler
class DeferredTool:
	'\n    Marker class for creating stub tools that defer actual work to agent post-processing.\n\n    Many tools in the agent system are "stubs" - they return their inputs or minimal data,\n    and the actual work (state injection, external calls, etc.) is done in the agent\'s\n    process_tool_output hook. This class makes that pattern explicit and self-documenting.\n\n    Usage:\n        tool = DeferredTool.create(\n            name="call_localization_agent",\n            description="Delegates to localization agent...",\n            args_schema=CallLocalizationAgentArgs,\n            returns_input_keys=["instructions"],\n            processing_note="Agent invokes localization orchestrator with instructions"\n        )\n    '
	@staticmethod
	def create(name:str,description:str,args_schema:Type[BaseModel],returns_input_keys:Optional[List[str]]=_A,returns_static:Optional[Dict[str,Any]]=_A,processing_note:Optional[str]=_A,handle_tool_error:Optional[Callable]=_A)->StructuredTool:
		"\n        Create a stub tool that returns its inputs for agent-side processing.\n\n        The tool function itself does minimal work - it either echoes back specified\n        input keys, returns a static value, or returns all inputs. The actual business\n        logic is implemented in the agent's process_tool_output hook.\n\n        Args:\n            name: Tool name used for registration and LLM tool calls\n            description: Tool description shown to the LLM\n            args_schema: Pydantic schema for tool arguments\n            returns_input_keys: List of input keys to echo back. If None and\n                               returns_static is None, echoes all inputs.\n            returns_static: Static dict to return instead of inputs.\n                           Takes precedence over returns_input_keys.\n            processing_note: Human-readable note explaining what the agent hook does.\n                            Included in the function docstring for documentation.\n            handle_tool_error: Optional error handler. Defaults to ToolExceptionHandler.\n\n        Returns:\n            A StructuredTool configured as a deferred stub\n        ";E=processing_note;D=returns_static;C=returns_input_keys;B=handle_tool_error
		if B is _A:B=ToolExceptionHandler.handle_error
		F=['DEFERRED TOOL: Returns inputs for agent-side processing.']
		if E:F.append(f"Agent hook: {E}")
		G='\n'.join(F)
		if D is not _A:
			def A(**A)->Dict[str,Any]:return dict(D)
		elif C is not _A:
			def A(**B)->Dict[str,Any]:return{A:B[A]for A in C if A in B}
		else:
			def A(**A)->Dict[str,Any]:return dict(A)
		A.__doc__=G;A.__name__=f"_deferred_{name}";return StructuredTool.from_function(func=A,name=name,description=description,args_schema=args_schema,handle_tool_error=B)
	@staticmethod
	def create_no_args(name:str,description:str,returns_static:Optional[Dict[str,Any]]=_A,processing_note:Optional[str]=_A,handle_tool_error:Optional[Callable]=_A)->StructuredTool:
		'\n        Create a stub tool with no arguments that returns a static value.\n\n        Convenience method for tools like compile_and_execute_test that take no\n        arguments and return an empty dict or simple marker value.\n\n        Args:\n            name: Tool name\n            description: Tool description for LLM\n            returns_static: Static dict to return (default: empty dict)\n            processing_note: Note explaining what the agent hook does\n            handle_tool_error: Optional error handler\n\n        Returns:\n            A StructuredTool configured as a no-args deferred stub\n        ';A=returns_static;from sakura.nl2test.models import NoArgs as B
		if A is _A:A={}
		return DeferredTool.create(name=name,description=description,args_schema=B,returns_static=A,processing_note=processing_note,handle_tool_error=handle_tool_error)