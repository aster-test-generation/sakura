from __future__ import annotations

import textwrap

from langchain_core.tools import StructuredTool

from nltest.nl2test.models import ViewTestCodeArgs, NoArgs
from nltest.nl2test.core.deferred_tool import DeferredTool
from nltest.nl2test.generation.common.tool_descriptions import (
    VIEW_TEST_CODE_DESC,
    COMPILE_AND_EXECUTE_TEST_DESC,
)
from nltest.utils.exceptions import ToolExceptionHandler


class CommonTestTools:
    """
    Shared test-related tools used by both supervisor and composition agents.

    These are deferred tools - they return minimal data and actual processing
    is done in the agent's process_tool_output hook.
    """

    @staticmethod
    def make_view_test_code_tool() -> StructuredTool:
        """
        Create the view_test_code tool.

        This is a deferred tool - it validates inputs and returns them.
        Actual file loading is done in the agent's process_tool_output hook.
        """
        def _view_test_code(start_line: int, end_line: int) -> dict:
            if start_line < 1 or end_line < 1:
                raise ValueError("start_line and end_line must be >= 1.")
            if end_line < start_line:
                raise ValueError("end_line must be >= start_line.")
            return {"start_line": start_line, "end_line": end_line}

        _view_test_code.__doc__ = (
            "DEFERRED TOOL: Returns line range for agent-side file loading.\n"
            "Agent hook loads test file and slices by line numbers."
        )

        return StructuredTool.from_function(
            func=_view_test_code,
            name="view_test_code",
            description=textwrap.dedent(VIEW_TEST_CODE_DESC).strip(),
            args_schema=ViewTestCodeArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    @staticmethod
    def make_compile_and_execute_test_tool() -> StructuredTool:
        """
        Create the compile_and_execute_test tool.

        This is a deferred tool with no arguments - returns empty dict.
        Actual Maven compilation and test execution is done in agent hook.
        """
        return DeferredTool.create_no_args(
            name="compile_and_execute_test",
            description=textwrap.dedent(COMPILE_AND_EXECUTE_TEST_DESC).strip(),
            returns_static={},
            processing_note="Agent runs Maven compilation and test execution using state.class_name and state.method_signature",
        )
