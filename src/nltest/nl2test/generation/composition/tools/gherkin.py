from __future__ import annotations

from langchain_core.tools import StructuredTool

from nltest.nl2test.models import LocalizedScenario, ModifyScenarioCommentArgs
from nltest.utils.exceptions import ToolExceptionHandler

from .base import BaseCompositionTools
from nltest.nl2test.generation.composition.tool_descriptions import (
    MODIFY_SCENARIO_COMMENT_DESC,
)


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
            description=MODIFY_SCENARIO_COMMENT_DESC,
            args_schema=ModifyScenarioCommentArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )
