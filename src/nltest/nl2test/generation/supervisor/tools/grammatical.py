from __future__ import annotations

from langchain_core.tools import StructuredTool

from .base import BaseSupervisorTools
from nltest.nl2test.models import AtomicBlockList, CallAgentGrammaticalArgs
from nltest.nl2test.generation.supervisor.tool_descriptions import (
    CALL_LOCALIZATION_AGENT_GRAMMATICAL_DESC,
    CALL_COMPOSITION_AGENT_GRAMMATICAL_DESC,
)
from nltest.utils.exceptions import ToolExceptionHandler


class GrammaticalSupervisorTools(BaseSupervisorTools):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.tools.append(self._make_call_localization_agent_tool())
        self.tools.append(self._make_call_composition_agent_tool())

    def _make_call_localization_agent_tool(self) -> StructuredTool:
        def _call_localization_agent(
            blocks: AtomicBlockList, instructions: str
        ) -> dict:
            # Defer the actual invocation to the Supervisor agent. Return inputs.
            return {"blocks": blocks, "instructions": instructions}

        return StructuredTool.from_function(
            func=_call_localization_agent,
            name="call_localization_agent",
            description=CALL_LOCALIZATION_AGENT_GRAMMATICAL_DESC,
            args_schema=CallAgentGrammaticalArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_call_composition_agent_tool(self) -> StructuredTool:
        def _call_composition_agent(blocks: AtomicBlockList, instructions: str) -> dict:
            # Defer the actual invocation to the Supervisor agent. Return inputs.
            return {"blocks": blocks, "instructions": instructions}

        return StructuredTool.from_function(
            func=_call_composition_agent,
            name="call_composition_agent",
            description=CALL_COMPOSITION_AGENT_GRAMMATICAL_DESC,
            args_schema=CallAgentGrammaticalArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )
