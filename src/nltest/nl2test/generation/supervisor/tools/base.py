from __future__ import annotations

from pathlib import Path
import textwrap
from typing import List, Tuple, Union

from langchain_core.tools import BaseTool

from nltest.nl2test.core.deferred_tool import DeferredTool
from nltest.nl2test.models import NoArgs, ViewTestCodeArgs
from nltest.utils.llm import LLMClient
from nltest.nl2test.generation.supervisor.tool_descriptions import FINALIZE_DESC
from nltest.nl2test.generation.common.tools.common_test_tools import CommonTestTools


class BaseSupervisorTools:
    """
    Base tool builder for supervisor agent.

    Provides common tools shared across decomposition modes:
    - view_test_code: View generated test source (deferred to agent)
    - compile_and_execute_test: Compile and run tests (deferred to agent)
    - finalize: End supervision

    Subclasses add mode-specific delegation tools.
    """

    def __init__(
            self,
            *,
            llm: LLMClient,
            project_root: Union[str, Path],
    ) -> None:
        self.llm = llm
        self.project_root = Path(project_root)

        self.tools: List[BaseTool] = [
            CommonTestTools.make_view_test_code_tool(),
            CommonTestTools.make_compile_and_execute_test_tool(),
            self._make_finalize_tool(),
        ]
        self.allow_duplicate_tools: List[BaseTool] = [
            CommonTestTools.make_view_test_code_tool(),
            CommonTestTools.make_compile_and_execute_test_tool(),
        ]

    def all(self) -> Tuple[List[BaseTool], List[BaseTool]]:
        return self.tools, self.allow_duplicate_tools

    def _make_finalize_tool(self) -> BaseTool:
        """Create the finalize tool to end supervision."""
        return DeferredTool.create_no_args(
            name="finalize",
            description=textwrap.dedent(FINALIZE_DESC).strip(),
            returns_static={"status": "finalize"},
            processing_note="Agent sets finalize_called=True and ends the run",
        )
