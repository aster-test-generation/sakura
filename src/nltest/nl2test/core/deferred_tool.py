from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from nltest.utils.exceptions import ToolExceptionHandler


class DeferredTool:
    """
    Marker class for creating stub tools that defer actual work to agent post-processing.

    Many tools in the agent system are "stubs" - they return their inputs or minimal data,
    and the actual work (state injection, external calls, etc.) is done in the agent's
    process_tool_output hook. This class makes that pattern explicit and self-documenting.

    Usage:
        tool = DeferredTool.create(
            name="call_localization_agent",
            description="Delegates to localization agent...",
            args_schema=CallLocalizationAgentArgs,
            returns_input_keys=["instructions"],
            processing_note="Agent invokes localization orchestrator with instructions"
        )
    """

    @staticmethod
    def create(
        name: str,
        description: str,
        args_schema: Type[BaseModel],
        returns_input_keys: Optional[List[str]] = None,
        returns_static: Optional[Dict[str, Any]] = None,
        processing_note: Optional[str] = None,
        handle_tool_error: Optional[Callable] = None,
    ) -> StructuredTool:
        """
        Create a stub tool that returns its inputs for agent-side processing.

        The tool function itself does minimal work - it either echoes back specified
        input keys, returns a static value, or returns all inputs. The actual business
        logic is implemented in the agent's process_tool_output hook.

        Args:
            name: Tool name used for registration and LLM tool calls
            description: Tool description shown to the LLM
            args_schema: Pydantic schema for tool arguments
            returns_input_keys: List of input keys to echo back. If None and
                               returns_static is None, echoes all inputs.
            returns_static: Static dict to return instead of inputs.
                           Takes precedence over returns_input_keys.
            processing_note: Human-readable note explaining what the agent hook does.
                            Included in the function docstring for documentation.
            handle_tool_error: Optional error handler. Defaults to ToolExceptionHandler.

        Returns:
            A StructuredTool configured as a deferred stub
        """
        if handle_tool_error is None:
            handle_tool_error = ToolExceptionHandler.handle_error

        doc_parts = ["DEFERRED TOOL: Returns inputs for agent-side processing."]
        if processing_note:
            doc_parts.append(f"Agent hook: {processing_note}")

        docstring = "\n".join(doc_parts)

        if returns_static is not None:

            def stub_func(**kwargs) -> Dict[str, Any]:
                return dict(returns_static)
        elif returns_input_keys is not None:

            def stub_func(**kwargs) -> Dict[str, Any]:
                return {k: kwargs[k] for k in returns_input_keys if k in kwargs}
        else:

            def stub_func(**kwargs) -> Dict[str, Any]:
                return dict(kwargs)

        stub_func.__doc__ = docstring
        stub_func.__name__ = f"_deferred_{name}"

        return StructuredTool.from_function(
            func=stub_func,
            name=name,
            description=description,
            args_schema=args_schema,
            handle_tool_error=handle_tool_error,
        )

    @staticmethod
    def create_no_args(
        name: str,
        description: str,
        returns_static: Optional[Dict[str, Any]] = None,
        processing_note: Optional[str] = None,
        handle_tool_error: Optional[Callable] = None,
    ) -> StructuredTool:
        """
        Create a stub tool with no arguments that returns a static value.

        Convenience method for tools like compile_and_execute_test that take no
        arguments and return an empty dict or simple marker value.

        Args:
            name: Tool name
            description: Tool description for LLM
            returns_static: Static dict to return (default: empty dict)
            processing_note: Note explaining what the agent hook does
            handle_tool_error: Optional error handler

        Returns:
            A StructuredTool configured as a no-args deferred stub
        """
        from nltest.nl2test.models import NoArgs

        if returns_static is None:
            returns_static = {}

        return DeferredTool.create(
            name=name,
            description=description,
            args_schema=NoArgs,
            returns_static=returns_static,
            processing_note=processing_note,
            handle_tool_error=handle_tool_error,
        )
