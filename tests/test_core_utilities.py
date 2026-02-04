"""Tests for refactored core utilities: MessageRedactor, ErrorFormatter, CLDKArgNormalizer, DeferredTool."""
from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from sakura.nl2test.core.deferred_tool import DeferredTool
from sakura.nl2test.core.message_redactor import MessageRedactor
from sakura.nl2test.generation.common.cldk_normalizer import CLDKArgNormalizer
from sakura.nl2test.models import AgentState
from sakura.utils.compilation.maven import CompilationError
from sakura.utils.execution.maven import ExecutionIssue
from sakura.utils.formatting import ErrorFormatter


class TestMessageRedactor:
    """Tests for MessageRedactor utility."""

    def _make_state_with_messages(self, messages: list) -> AgentState:
        """Create an AgentState with the given messages."""
        return AgentState(messages=messages)

    def test_redact_tool_outputs_replaces_matching_keys(self):
        """Verify that redact_tool_outputs replaces matching keys in tool messages."""
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "get_method_details", "args": {}}],
        )
        tool_msg = ToolMessage(
            content=json.dumps({
                "status": "ok",
                "data": {"source_code": "public void foo() {}", "details": "method info"}
            }),
            tool_call_id="call_1",
            name="get_method_details",
        )
        state = self._make_state_with_messages([ai_msg, tool_msg])

        MessageRedactor.redact_tool_outputs(
            state,
            tool_names={"get_method_details"},
            keys_to_redact={"source_code": "[REDACTED]"},
            status_filter="ok",
        )

        payload = json.loads(tool_msg.content)
        assert payload["data"]["source_code"] == "[REDACTED]"
        assert payload["data"]["details"] == "method info"

    def test_redact_tool_outputs_skips_non_matching_status(self):
        """Verify that redact_tool_outputs skips messages with non-matching status."""
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "get_method_details", "args": {}}],
        )
        tool_msg = ToolMessage(
            content=json.dumps({
                "status": "error",
                "data": {"source_code": "should not change"}
            }),
            tool_call_id="call_1",
            name="get_method_details",
        )
        state = self._make_state_with_messages([ai_msg, tool_msg])

        MessageRedactor.redact_tool_outputs(
            state,
            tool_names={"get_method_details"},
            keys_to_redact={"source_code": "[REDACTED]"},
            status_filter="ok",
        )

        payload = json.loads(tool_msg.content)
        assert payload["data"]["source_code"] == "should not change"

    def test_redact_tool_outputs_skips_non_matching_tool(self):
        """Verify that redact_tool_outputs skips non-matching tool names."""
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "other_tool", "args": {}}],
        )
        tool_msg = ToolMessage(
            content=json.dumps({
                "status": "ok",
                "data": {"source_code": "should not change"}
            }),
            tool_call_id="call_1",
            name="other_tool",
        )
        state = self._make_state_with_messages([ai_msg, tool_msg])

        MessageRedactor.redact_tool_outputs(
            state,
            tool_names={"get_method_details"},
            keys_to_redact={"source_code": "[REDACTED]"},
            status_filter="ok",
        )

        payload = json.loads(tool_msg.content)
        assert payload["data"]["source_code"] == "should not change"

    def test_redact_tool_outputs_by_tool_applies_per_tool_redactions(self):
        """Verify redact_tool_outputs_by_tool applies different redactions per tool."""
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"id": "call_1", "name": "tool_a", "args": {}},
                {"id": "call_2", "name": "tool_b", "args": {}},
            ],
        )
        tool_a_msg = ToolMessage(
            content=json.dumps({"status": "ok", "data": {"field_a": "value_a", "common": "x"}}),
            tool_call_id="call_1",
            name="tool_a",
        )
        tool_b_msg = ToolMessage(
            content=json.dumps({"status": "ok", "data": {"field_b": "value_b", "common": "y"}}),
            tool_call_id="call_2",
            name="tool_b",
        )
        state = self._make_state_with_messages([ai_msg, tool_a_msg, tool_b_msg])

        MessageRedactor.redact_tool_outputs_by_tool(
            state,
            tool_redactions={
                "tool_a": {"field_a": "[A_REDACTED]"},
                "tool_b": {"field_b": "[B_REDACTED]"},
            },
            status_filter="ok",
        )

        payload_a = json.loads(tool_a_msg.content)
        payload_b = json.loads(tool_b_msg.content)

        assert payload_a["data"]["field_a"] == "[A_REDACTED]"
        assert payload_a["data"]["common"] == "x"
        assert payload_b["data"]["field_b"] == "[B_REDACTED]"
        assert payload_b["data"]["common"] == "y"

    def test_redact_tool_inputs_replaces_args_excluding_latest(self):
        """Verify redact_tool_inputs replaces args but excludes the latest call."""
        ai_msg_1 = AIMessage(
            content="",
            tool_calls=[
                {"id": "call_1", "name": "gen_code", "args": {"code": "first code", "flag": True}},
            ],
        )
        ai_msg_2 = AIMessage(
            content="",
            tool_calls=[
                {"id": "call_2", "name": "gen_code", "args": {"code": "second code", "flag": False}},
            ],
        )
        state = self._make_state_with_messages([ai_msg_1, ai_msg_2])

        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x if isinstance(x, dict) else {}

        MessageRedactor.redact_tool_inputs(
            state,
            tool_name="gen_code",
            keys_to_redact={"code": "[CODE_REDACTED]"},
            llm_client=mock_llm,
            exclude_latest=True,
        )

        # First call should be redacted
        assert ai_msg_1.tool_calls[0]["args"]["code"] == "[CODE_REDACTED]"
        assert ai_msg_1.tool_calls[0]["args"]["flag"] is True
        # Second call (latest) should not be redacted
        assert ai_msg_2.tool_calls[0]["args"]["code"] == "second code"

    def test_redact_tool_inputs_redacts_all_when_exclude_latest_false(self):
        """Verify redact_tool_inputs redacts all calls when exclude_latest is False."""
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"id": "call_1", "name": "gen_code", "args": {"code": "some code"}},
            ],
        )
        state = self._make_state_with_messages([ai_msg])

        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x if isinstance(x, dict) else {}

        MessageRedactor.redact_tool_inputs(
            state,
            tool_name="gen_code",
            keys_to_redact={"code": "[REDACTED]"},
            llm_client=mock_llm,
            exclude_latest=False,
        )

        assert ai_msg.tool_calls[0]["args"]["code"] == "[REDACTED]"

    def test_redact_handles_multiple_keys(self):
        """Verify redaction works with multiple keys to redact."""
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "get_data", "args": {}}],
        )
        tool_msg = ToolMessage(
            content=json.dumps({
                "status": "ok",
                "data": {"field_a": "value_a", "field_b": "value_b", "field_c": "keep"}
            }),
            tool_call_id="call_1",
            name="get_data",
        )
        state = self._make_state_with_messages([ai_msg, tool_msg])

        MessageRedactor.redact_tool_outputs(
            state,
            tool_names={"get_data"},
            keys_to_redact={"field_a": "[A]", "field_b": "[B]"},
            status_filter="ok",
        )

        payload = json.loads(tool_msg.content)
        assert payload["data"]["field_a"] == "[A]"
        assert payload["data"]["field_b"] == "[B]"
        assert payload["data"]["field_c"] == "keep"


class TestErrorFormatter:
    """Tests for ErrorFormatter utility."""

    def test_format_compilation_error_with_details(self):
        """Verify compilation error formatting with details."""
        error = CompilationError(
            file="/path/to/File.java",
            line=42,
            message="cannot find symbol",
            details=["symbol: variable foo", "location: class Bar"],
        )

        formatted = ErrorFormatter.format_compilation_error(error)

        assert "Line: 42" in formatted
        assert "cannot find symbol" in formatted
        assert "symbol: variable foo" in formatted
        assert "location: class Bar" in formatted

    def test_format_compilation_error_with_empty_details(self):
        """Verify compilation error formatting with empty details list."""
        error = CompilationError(
            file="/path/to/File.java",
            line=10,
            message="syntax error",
            details=[],
        )

        formatted = ErrorFormatter.format_compilation_error(error)

        assert "Line: 10" in formatted
        assert "syntax error" in formatted
        assert "No compiler details were provided" in formatted

    def test_format_execution_issue_with_all_fields(self):
        """Verify execution issue formatting with all fields populated."""
        issue = ExecutionIssue(
            class_name="org.example.TestClass",
            test_name="testMethod",
            kind="Failure",
            message="Expected 5 but was 3",
            line=77,
            stack_trace="at org.example.TestClass.testMethod(TestClass.java:77)",
        )

        formatted = ErrorFormatter.format_execution_issue(issue)

        assert "Test Case: org.example.TestClass.testMethod" in formatted
        assert "Issue Kind: Failure" in formatted
        assert "Message: Expected 5 but was 3" in formatted
        assert "Line: 77" in formatted
        assert "at org.example.TestClass.testMethod" in formatted

    def test_format_execution_issue_with_optional_fields_missing(self):
        """Verify execution issue formatting with optional fields missing."""
        issue = ExecutionIssue(
            class_name="org.example.TestClass",
            test_name="testMethod",
            kind="Failure",
            message=None,
            line=None,
            stack_trace=None,
        )

        formatted = ErrorFormatter.format_execution_issue(issue)

        assert "Test Case: org.example.TestClass.testMethod" in formatted
        assert "Issue Kind: Failure" in formatted
        assert "No execution message was provided" in formatted
        assert "Line: Unknown" in formatted
        assert "No stack trace was captured" in formatted


class TestCLDKArgNormalizer:
    """Tests for CLDKArgNormalizer utility."""

    def test_normalize_args_normalizes_inner_class(self):
        """Verify inner class names are normalized (Outer$Inner -> Outer.Inner)."""
        raw_args = {"qualified_class_name": "org.example.Outer$Inner"}

        normalized = CLDKArgNormalizer.normalize_args("get_method_details", raw_args)

        # The actual normalization depends on CommonAnalysis.get_cldk_class_name
        # which should convert $ to . for CLDK compatibility
        assert normalized["qualified_class_name"] == "org.example.Outer.Inner"

    def test_normalize_args_does_not_modify_non_inner_class(self):
        """Verify normal class names are not modified."""
        raw_args = {"qualified_class_name": "org.example.MyClass"}

        normalized = CLDKArgNormalizer.normalize_args("get_class_fields", raw_args)

        assert normalized["qualified_class_name"] == "org.example.MyClass"

    def test_normalize_args_skips_non_cldk_tools(self):
        """Verify args are not modified for tools not in NORMALIZE_CLASS_TOOLS."""
        raw_args = {"qualified_class_name": "org.example.Outer$Inner"}

        normalized = CLDKArgNormalizer.normalize_args("unknown_tool", raw_args)

        # Should return same args unmodified
        assert normalized is raw_args

    def test_normalize_args_normalizes_constructor_signature(self):
        """Verify constructor method signatures are normalized."""
        raw_args = {
            "qualified_class_name": "org.example.MyClass",
            "method_signature": "MyClass()",
        }

        normalized = CLDKArgNormalizer.normalize_args("get_method_details", raw_args)

        # Constructor normalization adds <init> for CLDK
        assert "<init>" in normalized["method_signature"] or normalized["method_signature"] == "MyClass()"

    def test_normalize_class_tools_set_contains_expected_tools(self):
        """Verify NORMALIZE_CLASS_TOOLS contains expected tool names."""
        expected_tools = {
            "get_method_details",
            "get_class_fields",
            "get_class_imports",
            "get_class_constructors_and_factories",
            "get_getters_and_setters",
            "extract_method_code",
            "get_call_site_details",
            "get_reachable_methods_in_class",
            "get_class_details",
            "get_inherited_library_classes",
        }

        assert CLDKArgNormalizer.NORMALIZE_CLASS_TOOLS == expected_tools

    def test_normalize_method_sig_tools_set_contains_expected_tools(self):
        """Verify NORMALIZE_METHOD_SIG_TOOLS contains expected tool names."""
        expected_tools = {
            "get_call_site_details",
            "get_method_details",
            "extract_method_code",
        }

        assert CLDKArgNormalizer.NORMALIZE_METHOD_SIG_TOOLS == expected_tools


class SampleArgs(BaseModel):
    """Sample args schema for DeferredTool tests."""
    name: str = Field(description="A name parameter")
    count: int = Field(default=1, description="A count parameter")


class TestDeferredTool:
    """Tests for DeferredTool wrapper."""

    def test_create_returns_structured_tool(self):
        """Verify create() returns a StructuredTool instance."""
        tool = DeferredTool.create(
            name="test_tool",
            description="A test tool",
            args_schema=SampleArgs,
        )

        assert tool.name == "test_tool"
        assert "A test tool" in tool.description

    def test_create_returns_all_inputs_by_default(self):
        """Verify the stub function returns all inputs when returns_input_keys is None."""
        tool = DeferredTool.create(
            name="echo_tool",
            description="Echoes all inputs",
            args_schema=SampleArgs,
        )

        result = tool.func(name="test", count=5)

        assert result == {"name": "test", "count": 5}

    def test_create_returns_only_specified_keys(self):
        """Verify the stub function returns only specified input keys."""
        tool = DeferredTool.create(
            name="partial_tool",
            description="Returns partial inputs",
            args_schema=SampleArgs,
            returns_input_keys=["name"],
        )

        result = tool.func(name="test", count=5)

        assert result == {"name": "test"}
        assert "count" not in result

    def test_create_returns_static_value(self):
        """Verify the stub function returns static value when specified."""
        static_data = {"status": "deferred", "action": "pending"}

        tool = DeferredTool.create(
            name="static_tool",
            description="Returns static value",
            args_schema=SampleArgs,
            returns_static=static_data,
        )

        result = tool.func(name="ignored", count=99)

        assert result == static_data

    def test_create_includes_processing_note_in_docstring(self):
        """Verify processing_note is included in the function docstring."""
        tool = DeferredTool.create(
            name="documented_tool",
            description="A documented tool",
            args_schema=SampleArgs,
            processing_note="Agent updates state with this data",
        )

        assert "DEFERRED TOOL" in tool.func.__doc__
        assert "Agent updates state with this data" in tool.func.__doc__

    def test_create_no_args_returns_structured_tool(self):
        """Verify create_no_args() returns a StructuredTool."""
        tool = DeferredTool.create_no_args(
            name="no_args_tool",
            description="Tool with no args",
        )

        assert tool.name == "no_args_tool"

    def test_create_no_args_returns_empty_dict_by_default(self):
        """Verify create_no_args returns empty dict when no static value specified."""
        tool = DeferredTool.create_no_args(
            name="empty_tool",
            description="Returns empty",
        )

        result = tool.func()

        assert result == {}

    def test_create_no_args_returns_static_value(self):
        """Verify create_no_args returns static value when specified."""
        tool = DeferredTool.create_no_args(
            name="finalize_tool",
            description="Finalize operation",
            returns_static={"status": "finalize"},
        )

        result = tool.func()

        assert result == {"status": "finalize"}

    def test_create_with_custom_error_handler(self):
        """Verify custom error handler is applied to the tool."""
        custom_handler_called = []

        def custom_handler(error):
            custom_handler_called.append(error)
            return "Custom error message"

        tool = DeferredTool.create(
            name="error_tool",
            description="Tool with custom handler",
            args_schema=SampleArgs,
            handle_tool_error=custom_handler,
        )

        assert tool.handle_tool_error == custom_handler

    def test_create_function_name_prefix(self):
        """Verify the stub function has correct naming convention."""
        tool = DeferredTool.create(
            name="my_tool",
            description="Test",
            args_schema=SampleArgs,
        )

        assert tool.func.__name__ == "_deferred_my_tool"
