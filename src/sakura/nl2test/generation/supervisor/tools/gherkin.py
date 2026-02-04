from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Union

from langchain_core.tools import BaseTool

from .base import BaseSupervisorTools
from sakura.nl2test.core.deferred_tool import DeferredTool
from sakura.nl2test.models import (
    CallLocalizationAgentGherkinArgs,
    CallCompositionAgentGherkinArgs,
)
from sakura.nl2test.generation.supervisor.tool_descriptions import (
    CALL_LOCALIZATION_AGENT_GHERKIN_DESC,
    CALL_COMPOSITION_AGENT_GHERKIN_DESC,
)
from sakura.utils.llm import LLMClient


class GherkinSupervisorTools(BaseSupervisorTools):
    """
    Tool builder for Gherkin-mode supervisor agent.

    Extends base tools with Gherkin-specific delegation tools:
    - call_localization_agent: Delegate to localization agent (deferred)
    - call_composition_agent: Delegate to composition agent (deferred)
    """

    def __init__(self, *, llm: LLMClient, project_root: Union[str, Path]) -> None:
        super().__init__(llm=llm, project_root=project_root)

        localization_tool = self._make_call_localization_agent_tool()
        composition_tool = self._make_call_composition_agent_tool()

        self.tools.append(localization_tool)
        self.tools.append(composition_tool)
        self.allow_duplicate_tools.append(localization_tool)
        self.allow_duplicate_tools.append(composition_tool)

    def _make_call_localization_agent_tool(self) -> BaseTool:
        """
        Create the call_localization_agent tool for Gherkin mode.

        This is a deferred tool - it returns the instructions and the agent's
        process_tool_output hook invokes the actual localization orchestrator.
        """
        return DeferredTool.create(
            name="call_localization_agent",
            description=textwrap.dedent(CALL_LOCALIZATION_AGENT_GHERKIN_DESC).strip(),
            args_schema=CallLocalizationAgentGherkinArgs,
            returns_input_keys=["instructions"],
            processing_note="Agent invokes localization orchestrator with instructions, "
            "injecting current LocalizedScenario from state",
        )

    def _make_call_composition_agent_tool(self) -> BaseTool:
        """
        Create the call_composition_agent tool for Gherkin mode.

        This is a deferred tool - it returns the instructions and the agent's
        process_tool_output hook invokes the actual composition orchestrator.
        """
        return DeferredTool.create(
            name="call_composition_agent",
            description=textwrap.dedent(CALL_COMPOSITION_AGENT_GHERKIN_DESC).strip(),
            args_schema=CallCompositionAgentGherkinArgs,
            returns_input_keys=["instructions"],
            processing_note="Agent invokes composition orchestrator with instructions, "
            "injecting current LocalizedScenario from state",
        )
