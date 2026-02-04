import textwrap
from typing import Tuple
from cldk.analysis.java import JavaAnalysis
from sakura.nl2test.preprocessing.searchers import MethodSearcher, ClassSearcher

from langchain_core.tools import StructuredTool

from sakura.nl2test.generation.localization.tool_descriptions import (
    FINALIZE_LOCALIZED_SCENARIO_DESC,
)
from sakura.nl2test.models import LocalizedScenario, FinalizeScenarioArgs
from sakura.utils.exceptions import ToolExceptionHandler
from sakura.utils.exceptions.tool_exceptions import BlockNotFoundError

from .base import BaseLocalizationTools


class GherkinLocalizationTools(BaseLocalizationTools):
    def __init__(
        self,
        *,
        analysis: JavaAnalysis,
        method_searcher: MethodSearcher,
        class_searcher: ClassSearcher,
    ) -> None:
        super().__init__(
            analysis=analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
        )
        # Add the gherkin finalize tool
        self.tools.append(self._make_finalize_tool())

    # Finalize the scenario and end the agent (Gherkin mode)
    def _make_finalize_tool(self) -> StructuredTool:
        def _finalize(
            localized_scenario: LocalizedScenario, comments: str
        ) -> Tuple[LocalizedScenario, str]:
            if localized_scenario is None:
                raise BlockNotFoundError(
                    "Current scenario not found",
                    extra_info={"localized_scenario": localized_scenario},
                )

            return localized_scenario, comments

        return StructuredTool.from_function(
            func=_finalize,
            name="finalize",
            description=textwrap.dedent(FINALIZE_LOCALIZED_SCENARIO_DESC).strip(),
            args_schema=FinalizeScenarioArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )
