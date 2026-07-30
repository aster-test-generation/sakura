from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sakura.nl2test.generation.supervisor.tools.base import BaseSupervisorTools
from sakura.nl2test.generation.supervisor.tools.gherkin import GherkinSupervisorTools
from sakura.nl2test.generation.supervisor.tools.grammatical import (
    GrammaticalSupervisorTools,
)
from sakura.nl2test.models import AgentState


class TestSupervisorToolInjection:
    """Tests verifying tool injection after refactoring to shared utilities."""

    def test_base_supervisor_tools_contains_expected_tools(self):
        """Verify BaseSupervisorTools creates expected base tools."""
        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tool_builder = BaseSupervisorTools(llm=mock_llm, project_root="/tmp/test")
        tools, allow_duplicates = tool_builder.all()

        tool_names = {t.name for t in tools}
        expected_base_tools = {"view_test_code", "compile_and_execute_test", "finalize"}

        assert expected_base_tools.issubset(tool_names), (
            f"Missing base tools: {expected_base_tools - tool_names}"
        )

    def test_gherkin_supervisor_tools_contains_all_tools(self):
        """Verify GherkinSupervisorTools includes base + Gherkin-specific tools."""
        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tool_builder = GherkinSupervisorTools(llm=mock_llm, project_root="/tmp/test")
        tools, allow_duplicates = tool_builder.all()

        tool_names = {t.name for t in tools}
        expected_tools = {
            "view_test_code",
            "compile_and_execute_test",
            "finalize",
            "call_localization_agent",
            "call_composition_agent",
        }

        assert expected_tools == tool_names, (
            f"Tool mismatch. Expected: {expected_tools}, Got: {tool_names}"
        )

    def test_grammatical_supervisor_tools_contains_all_tools(self):
        """Verify GrammaticalSupervisorTools includes base + Grammatical-specific tools."""
        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tool_builder = GrammaticalSupervisorTools(
            llm=mock_llm, project_root="/tmp/test"
        )
        tools, allow_duplicates = tool_builder.all()

        tool_names = {t.name for t in tools}
        expected_tools = {
            "view_test_code",
            "compile_and_execute_test",
            "finalize",
            "call_localization_agent",
            "call_composition_agent",
        }

        assert expected_tools == tool_names, (
            f"Tool mismatch. Expected: {expected_tools}, Got: {tool_names}"
        )

    def test_deferred_tool_call_localization_returns_instructions(self):
        """Verify call_localization_agent returns instructions as expected."""
        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tool_builder = GherkinSupervisorTools(llm=mock_llm, project_root="/tmp/test")
        tools, _ = tool_builder.all()

        call_loc_tool = next(t for t in tools if t.name == "call_localization_agent")
        result = call_loc_tool.func(instructions="Find relevant methods")

        assert result == {"instructions": "Find relevant methods"}

    def test_deferred_tool_call_composition_returns_instructions(self):
        """Verify call_composition_agent returns instructions as expected."""
        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tool_builder = GherkinSupervisorTools(llm=mock_llm, project_root="/tmp/test")
        tools, _ = tool_builder.all()

        call_comp_tool = next(t for t in tools if t.name == "call_composition_agent")
        result = call_comp_tool.func(instructions="Generate test code")

        assert result == {"instructions": "Generate test code"}

    def test_deferred_tool_compile_and_execute_returns_empty(self):
        """Verify compile_and_execute_test returns empty dict."""
        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tool_builder = BaseSupervisorTools(llm=mock_llm, project_root="/tmp/test")
        tools, _ = tool_builder.all()

        compile_tool = next(t for t in tools if t.name == "compile_and_execute_test")
        result = compile_tool.func()

        assert result == {}


class TestSupervisorForceEnd:
    """Tests for supervisor agent force_end behavior (NoArgs optimization)."""

    def test_execute_force_end_increments_attempts(self):
        """Verify _execute_force_end increments force_end_attempts each call."""
        from sakura.nl2test.generation.supervisor.agent import SupervisorReActAgent

        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tools, _ = BaseSupervisorTools(llm=mock_llm, project_root="/tmp/test").all()
        agent = SupervisorReActAgent(
            llm=mock_llm,
            tools=tools,
            system_message="Test system message",
            nl_description="Test description",
            project_root="/tmp/test",
        )

        state = AgentState()
        state.force_end_attempts = 2

        result_state = agent._execute_force_end(state)

        assert result_state.force_end_attempts == 3

    def test_force_finalize_prompt_methods_raise_not_implemented(self):
        """Verify prompt methods raise NotImplementedError since they're unused."""
        from sakura.nl2test.generation.supervisor.agent import SupervisorReActAgent

        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tools, _ = BaseSupervisorTools(llm=mock_llm, project_root="/tmp/test").all()
        agent = SupervisorReActAgent(
            llm=mock_llm,
            tools=tools,
            system_message="Test system message",
            nl_description="Test description",
            project_root="/tmp/test",
        )

        with pytest.raises(NotImplementedError):
            agent._get_force_finalize_system_prompt()

        with pytest.raises(NotImplementedError):
            agent._get_force_finalize_chat_prompt()

        with pytest.raises(NotImplementedError):
            agent._process_force_finalize_result(MagicMock(), AgentState())
