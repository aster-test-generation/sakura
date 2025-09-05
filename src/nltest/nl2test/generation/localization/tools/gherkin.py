from typing import Tuple

from langchain_core.tools import StructuredTool

from nltest.nl2test.generation.localization.tool_descriptions import (
    FINALIZE_LOCALIZED_SCENARIO_DESC,
)
from nltest.nl2test.models import LocalizedScenario, FinalizeScenarioArgs
from nltest.utils.exceptions import ToolExceptionHandler
from nltest.utils.exceptions.tool_exceptions import BlockNotFoundError

from .base import BaseLocalizationTools


class GherkinLocalizationTools(BaseLocalizationTools):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Add the gherkin finalize tool
        self.tools.append(self._make_finalize_tool())

    # Finalize the scenario and end the agent (Gherkin mode)
    def _make_finalize_tool(self) -> StructuredTool:
        def _finalize(
            scenario: LocalizedScenario, comments: str
        ) -> Tuple[LocalizedScenario, str]:
            if scenario is None:
                raise BlockNotFoundError(
                    "Current scenario not found",
                    extra_info={"scenario": scenario},
                )

            return scenario, comments

        return StructuredTool.from_function(
            func=_finalize,
            name="finalize",
            description=FINALIZE_LOCALIZED_SCENARIO_DESC,
            args_schema=FinalizeScenarioArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )
