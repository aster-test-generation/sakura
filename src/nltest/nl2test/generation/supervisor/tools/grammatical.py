from __future__ import annotations

from pathlib import Path
from typing import Union
from langchain_core.tools import StructuredTool

from .base import BaseSupervisorTools
from nltest.nl2test.models import CallAgentGrammaticalArgs
from nltest.nl2test.generation.supervisor.tool_descriptions import (
    CALL_LOCALIZATION_AGENT_GRAMMATICAL_DESC,
    CALL_COMPOSITION_AGENT_GRAMMATICAL_DESC,
)
from nltest.utils.exceptions import ToolExceptionHandler
from nltest.utils.llm import LLMClient


class GrammaticalSupervisorTools(BaseSupervisorTools):
    def __init__(self, *, llm: LLMClient, project_root: Union[str, Path]) -> None:
        super().__init__(llm=llm, project_root=project_root)

        self.tools.append(self._make_call_localization_agent_tool())
        self.tools.append(self._make_call_composition_agent_tool())

    def _make_call_localization_agent_tool(self) -> StructuredTool:
        def _call_localization_agent(instructions: str) -> dict:
            # Defer the actual invocation to the Supervisor agent. Return inputs.
            return {"instructions": instructions}

        return StructuredTool.from_function(
            func=_call_localization_agent,
            name="call_localization_agent",
            description=CALL_LOCALIZATION_AGENT_GRAMMATICAL_DESC,
            args_schema=CallAgentGrammaticalArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_call_composition_agent_tool(self) -> StructuredTool:
        def _call_composition_agent(instructions: str) -> dict:
            # Defer the actual invocation to the Supervisor agent. Return inputs.
            return {"instructions": instructions}

        return StructuredTool.from_function(
            func=_call_composition_agent,
            name="call_composition_agent",
            description=CALL_COMPOSITION_AGENT_GRAMMATICAL_DESC,
            args_schema=CallAgentGrammaticalArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )
