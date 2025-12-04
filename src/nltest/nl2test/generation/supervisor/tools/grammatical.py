from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Union

from langchain_core.tools import BaseTool

from .base import BaseSupervisorTools
from nltest.nl2test.core.deferred_tool import DeferredTool
from nltest.nl2test.models import CallAgentGrammaticalArgs
from nltest.nl2test.generation.supervisor.tool_descriptions import (
    CALL_LOCALIZATION_AGENT_GRAMMATICAL_DESC,
    CALL_COMPOSITION_AGENT_GRAMMATICAL_DESC,
)
from nltest.utils.llm import LLMClient


class GrammaticalSupervisorTools(BaseSupervisorTools):
    """
    Tool builder for Grammatical-mode supervisor agent.

    Extends base tools with Grammatical-specific delegation tools:
    - call_localization_agent: Delegate to localization agent (deferred)
    - call_composition_agent: Delegate to composition agent (deferred)
    """

    def __init__(self, *, llm: LLMClient, project_root: Union[str, Path]) -> None:
        super().__init__(llm=llm, project_root=project_root)

        self.tools.append(self._make_call_localization_agent_tool())
        self.tools.append(self._make_call_composition_agent_tool())

    def _make_call_localization_agent_tool(self) -> BaseTool:
        """
        Create the call_localization_agent tool for Grammatical mode.

        This is a deferred tool - it returns the instructions and the agent's
        process_tool_output hook invokes the actual localization orchestrator.
        """
        return DeferredTool.create(
            name="call_localization_agent",
            description=textwrap.dedent(CALL_LOCALIZATION_AGENT_GRAMMATICAL_DESC).strip(),
            args_schema=CallAgentGrammaticalArgs,
            returns_input_keys=["instructions"],
            processing_note="Agent invokes localization orchestrator with instructions, "
                           "injecting current AtomicBlockList from state",
        )

    def _make_call_composition_agent_tool(self) -> BaseTool:
        """
        Create the call_composition_agent tool for Grammatical mode.

        This is a deferred tool - it returns the instructions and the agent's
        process_tool_output hook invokes the actual composition orchestrator.
        """
        return DeferredTool.create(
            name="call_composition_agent",
            description=textwrap.dedent(CALL_COMPOSITION_AGENT_GRAMMATICAL_DESC).strip(),
            args_schema=CallAgentGrammaticalArgs,
            returns_input_keys=["instructions"],
            processing_note="Agent invokes composition orchestrator with instructions, "
                           "injecting current AtomicBlockList from state",
        )
