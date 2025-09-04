from __future__ import annotations

from .base import BaseCompositionTools
from langchain_core.tools import StructuredTool

from nltest.nl2test.models import LocalizedScenario
from nltest.nl2test.models.agents import ModifyScenarioCommentArgs
from nltest.utils.exceptions import ToolExceptionHandler


class GherkinCompositionTools(BaseCompositionTools):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tools.append(self._make_modify_scenario_comment_tool())

    def _make_modify_scenario_comment_tool(self) -> StructuredTool:
        def _modify_scenario_comment(id: int, comment: str):
            return id, comment

        return StructuredTool.from_function(
            func=_modify_scenario_comment,
            name="modify_scenario_comment",
            description="",
            args_schema=ModifyScenarioCommentArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )
