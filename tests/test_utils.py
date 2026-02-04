"""Tests for tool error handling in ToolExceptionHandler and ReActAgent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from langchain_core.messages import ToolMessage

from sakura.nl2test.core.deferred_tool import DeferredTool
from sakura.nl2test.core.react_agent import ReActAgent
from sakura.nl2test.models import AgentState
from sakura.utils.exceptions import InvalidArgumentError, ToolExceptionHandler


class TestToolErrorHandling:
    """Tests for tool error envelope formatting."""

    def test_tool_exception_handler_builds_error_envelope(self) -> None:
        """Verify ToolExceptionHandler returns a structured error envelope."""
        error = InvalidArgumentError("Bad input", extra_info={"field": "value"})

        payload = ToolExceptionHandler.handle_error(error)

        assert payload["status"] == "error"
        assert payload["error"]["code"] == "InvalidArgumentError"
        assert "Bad input" in payload["error"]["message"]
        assert payload["error"]["details"]["field"] == "value"

    def test_react_agent_formats_error_envelope(self) -> None:
        """Verify ReActAgent turns tool error envelopes into error messages."""
        error = InvalidArgumentError("Bad input", extra_info={"field": "value"})
        tool_error = ToolExceptionHandler.handle_error(error)

        finalize_tool = DeferredTool.create_no_args(
            name="finalize",
            description="Finalize tool",
        )
        agent = ReActAgent(
            llm=MagicMock(),
            tools=[finalize_tool],
            allow_parallelize=False,
        )
        state = AgentState()
        outputs: list[ToolMessage] = []

        agent.process_tool_output(
            tool_call={"id": "call_1", "name": "sample_tool"},
            result=tool_error,
            state=state,
            outputs=outputs,
        )

        assert len(outputs) == 1
        payload = json.loads(outputs[0].content)
        assert payload["status"] == "error"
        assert payload["error"]["code"] == "InvalidArgumentError"
        assert payload["error"]["details"]["field"] == "value"
        assert payload["error"]["details"]["tool"] == "sample_tool"
        assert payload["error"]["details"]["tool_call_id"] == "call_1"
