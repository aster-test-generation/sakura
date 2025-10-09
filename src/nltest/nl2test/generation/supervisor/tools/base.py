from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

from langchain_core.tools import BaseTool, StructuredTool

from nltest.nl2test.core.react_agent import ReActAgent
from nltest.nl2test.models import NL2TestInput
from nltest.utils.llm import LLMClient
from nltest.nl2test.generation.supervisor.tool_descriptions import (
    VIEW_TEST_CODE_DESC,
    COMPILE_AND_EXECUTE_TEST_DESC,
    FINALIZE_DESC,
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
        def _view_test_code() -> dict:
            return {}

        return StructuredTool.from_function(
            func=_view_test_code,
            name="view_test_code",
            description=VIEW_TEST_CODE_DESC,
        )

    def _make_compile_and_execute_code_tool(self) -> StructuredTool:
        def _compile_and_execute_test() -> dict:
            return {}

        return StructuredTool.from_function(
            func=_compile_and_execute_test,
            name="compile_and_execute_test",
            description=COMPILE_AND_EXECUTE_TEST_DESC,
        )

    def _make_finalize_tool(self) -> StructuredTool:
        def _finalize() -> str:
            return "finalize"

        return StructuredTool.from_function(
            func=_finalize,
            name="finalize",
            description=FINALIZE_DESC,
        )
