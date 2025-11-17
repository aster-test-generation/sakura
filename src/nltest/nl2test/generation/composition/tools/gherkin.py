from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Union
from langchain_core.tools import StructuredTool

from nltest.nl2test.models import LocalizedScenario, ModifyScenarioCommentArgs
from nltest.utils.exceptions import ToolExceptionHandler

from .base import BaseCompositionTools
from cldk.analysis.java import JavaAnalysis
from nltest.nl2test.preprocessing.searchers import MethodSearcher, ClassSearcher
from nltest.utils.llm import LLMClient
from nltest.nl2test.models import NL2TestInput
from nltest.nl2test.generation.composition.tool_descriptions import (
    MODIFY_SCENARIO_COMMENT_DESC,
)


class GherkinCompositionTools(BaseCompositionTools):
    def __init__(
        self,
        *,
        analysis: JavaAnalysis,
        method_searcher: MethodSearcher,
        class_searcher: ClassSearcher,
        structured_llm: LLMClient,
        project_root: Union[str, Path],
        nl2_input: NL2TestInput,
    ):
        super().__init__(
            analysis=analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            structured_llm=structured_llm,
            project_root=str(project_root),
            nl2_input=nl2_input,
        )
        self.tools.append(self._make_modify_scenario_comment_tool())

    def _make_modify_scenario_comment_tool(self) -> StructuredTool:
        def _modify_scenario_comment(id: int, comment: str):
            return id, comment

        return StructuredTool.from_function(
            func=_modify_scenario_comment,
            name="modify_scenario_comment",
            description=textwrap.dedent(MODIFY_SCENARIO_COMMENT_DESC).strip(),
            args_schema=ModifyScenarioCommentArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )
