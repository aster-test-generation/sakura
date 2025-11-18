from __future__ import annotations

from pathlib import Path
import textwrap
from typing import List, Tuple, Union

from langchain_core.tools import BaseTool, StructuredTool

from nltest.nl2test.core.react_agent import ReActAgent
from nltest.nl2test.models import NoArgs, ViewTestCodeArgs
from nltest.utils.llm import LLMClient
from nltest.nl2test.generation.supervisor.tool_descriptions import (
    FINALIZE_DESC,
)
from nltest.nl2test.generation.common.tool_descriptions import (
    VIEW_TEST_CODE_DESC,
    COMPILE_AND_EXECUTE_TEST_DESC,
)


class BaseSupervisorTools:
    def __init__(
            self,
            *,
            llm: LLMClient,
            project_root: Union[str, Path],
    ) -> None:
        self.llm = llm
        self.project_root = Path(project_root)

        self.tools: List[BaseTool] = [
            self._make_view_test_code_tool(),
            self._make_compile_and_execute_code_tool(),
            self._make_finalize_tool(),
        ]
        self.allow_duplicate_tools: List[BaseTool] = [
            self._make_view_test_code_tool(),
            self._make_compile_and_execute_code_tool(),
        ]

    def all(self) -> Tuple[List[BaseTool], List[BaseTool]]:
        return self.tools, self.allow_duplicate_tools

    # The following tools are simple stubs; behavior is implemented in the Supervisor agent
    def _make_view_test_code_tool(self) -> StructuredTool:
        def _view_test_code(start_line: int, end_line: int) -> dict:
            if start_line <= 1 or end_line <= 1:
                raise ValueError("start_line and end_line must be greater than 1.")
            if end_line < start_line:
                raise ValueError("end_line must be greater than or equal to start_line.")
            return {"start_line": start_line, "end_line": end_line}

        return StructuredTool.from_function(
            func=_view_test_code,
            name="view_test_code",
            description=textwrap.dedent(VIEW_TEST_CODE_DESC).strip(),
            args_schema=ViewTestCodeArgs,
        )

    def _make_compile_and_execute_code_tool(self) -> StructuredTool:
        def _compile_and_execute_test() -> dict:
            return {}

        return StructuredTool.from_function(
            func=_compile_and_execute_test,
            name="compile_and_execute_test",
            description=textwrap.dedent(COMPILE_AND_EXECUTE_TEST_DESC).strip(),
            args_schema=NoArgs,
        )

    def _make_finalize_tool(self) -> StructuredTool:
        def _finalize() -> str:
            # The agent will consume this and finalize the run state.
            return "finalize"

        return StructuredTool.from_function(
            func=_finalize,
            name="finalize",
            description=textwrap.dedent(FINALIZE_DESC).strip(),
            args_schema=NoArgs,
        )
