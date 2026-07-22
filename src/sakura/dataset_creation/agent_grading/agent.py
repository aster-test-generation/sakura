"""Claude Agent SDK wrapper for the description-grading agent.

Ported and trimmed from sakura-uncertainty (agents/base.py + agents/config.py):
one read-only agent role, structured output always on, git guard always on.
Container-side module: relative imports only, requires claude-agent-sdk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from claude_agent_sdk import (
    ClaudeAgentOptions,
    HookMatcher,
    ResultMessage,
    query,
)

from .prompt_renderer import PromptRenderer
from .schema import STRUCTURED_OUTPUT_FORMAT

GIT_COMMAND_PATTERN = re.compile(r"(^|[\s;&|(){}])git([\s;&|(){}]|$)")
GIT_PATH_PATTERN = re.compile(r"(^|[/\s])\.git([/\s]|$)")

THINKING_LEVELS = ("low", "medium", "high", "xhigh", "max")
PROMPT_DIR = "grader"

# Docker is the security boundary; inside it the agent runs unattended, so
# permissions are bypassed and the tool surface is pinned to read-only
# exploration. Grades are delivered via structured output, never file writes.
PERMISSION_MODE = "bypassPermissions"
ALLOWED_TOOLS = ["Read", "Grep", "Glob", "Bash"]
DISALLOWED_TOOLS = [
    "WebSearch",
    "WebFetch",
    "AskUserQuestion",
    "Write",
    "Edit",
    "NotebookEdit",
    "TodoWrite",
]


@dataclass
class AgentResult:
    """The outcome of a single agent run.

    Attributes:
        text: The agent's final assistant text.
        structured: Parsed structured output matching STRUCTURED_OUTPUT_FORMAT.
        raw_messages: The full SDK message stream, for artifacting/debugging.
        total_cost_usd: The SDK ``ResultMessage`` cost estimate (client-side;
            accurate only for model ids the CLI's price table recognizes).
        usage: The aggregate token-usage dict.
        model_usage: The per-model token/cost breakdown keyed by model id.
        num_turns: Number of agent turns in the run.
        duration_ms: Wall-clock duration of the run, in milliseconds.
        duration_api_ms: API time of the run, in milliseconds.
        stop_reason: Why the agent stopped (e.g. ``"end_turn"``).
    """

    text: str
    structured: Optional[Dict[str, Any]] = None
    raw_messages: List[Any] = field(default_factory=list)
    total_cost_usd: Optional[float] = None
    usage: Optional[Dict[str, Any]] = None
    model_usage: Optional[Dict[str, Any]] = None
    num_turns: Optional[int] = None
    duration_ms: Optional[int] = None
    duration_api_ms: Optional[int] = None
    stop_reason: Optional[str] = None


@dataclass
class GraderAgentConfig:
    """Configuration for the grading agent.

    Attributes:
        model: The Claude model id to use.
        thinking: SDK reasoning effort passed as ``effort`` (None to omit).
        extra_options: Extra keyword args splatted into ``ClaudeAgentOptions``.
            Keys must be valid ``ClaudeAgentOptions`` field names.
    """

    model: str
    thinking: Optional[str] = "high"
    extra_options: Dict[str, Any] = field(default_factory=dict)


class GradingAgent:
    """Thin wrapper around the Claude Agent SDK ``query()`` call."""

    def __init__(self, config: GraderAgentConfig, renderer: PromptRenderer) -> None:
        self.config = config
        self.renderer = renderer

    def _build_options(
        self, system_context: Dict[str, Any], cwd: Path
    ) -> ClaudeAgentOptions:
        options_kwargs: Dict[str, Any] = {
            "system_prompt": self.renderer.render(
                PROMPT_DIR, "system", system_context
            ),
            "model": self.config.model,
            "cwd": str(cwd),
            "allowed_tools": list(ALLOWED_TOOLS),
            "disallowed_tools": list(DISALLOWED_TOOLS),
            "permission_mode": PERMISSION_MODE,
            "setting_sources": [],
            "output_format": STRUCTURED_OUTPUT_FORMAT,
        }
        if self.config.thinking is not None:
            options_kwargs["effort"] = self.config.thinking
        options_kwargs.update(self.config.extra_options)
        options_kwargs["hooks"] = self._with_git_guard_hook(
            options_kwargs.get("hooks")
        )
        return ClaudeAgentOptions(**options_kwargs)

    def _with_git_guard_hook(
        self, hooks: Optional[Dict[str, List[HookMatcher]]]
    ) -> Dict[str, List[HookMatcher]]:
        merged_hooks: Dict[str, List[HookMatcher]] = {}
        for event_name, matchers in (hooks or {}).items():
            merged_hooks[event_name] = list(matchers)
        merged_hooks.setdefault("PreToolUse", []).append(
            HookMatcher(matcher=None, hooks=[self._block_git_tool_use])
        )
        return merged_hooks

    async def _block_git_tool_use(
        self,
        hook_input: Dict[str, Any],
        _tool_use_id: str | None,
        _context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if hook_input.get("hook_event_name") != "PreToolUse":
            return {}

        tool_name = str(hook_input.get("tool_name", ""))
        tool_input = hook_input.get("tool_input", {})
        if not isinstance(tool_input, dict):
            return {}

        if self._touches_git_path(tool_input):
            return self._blocked_tool_response(
                "The repository's .git metadata is reserved for the "
                "grading orchestrator."
            )
        if tool_name == "Bash" and self._is_git_command(tool_input):
            return self._blocked_tool_response(
                "Git commands are reserved for the grading orchestrator and "
                "may not be run by agents."
            )
        return {}

    @staticmethod
    def _blocked_tool_response(reason: str) -> Dict[str, Any]:
        return {
            "decision": "block",
            "reason": reason,
            "systemMessage": reason,
        }

    @staticmethod
    def _is_git_command(tool_input: Dict[str, Any]) -> bool:
        command = str(tool_input.get("command", ""))
        if GIT_COMMAND_PATTERN.search(command):
            return True
        blocked_markers = ("GIT_DIR", "GIT_WORK_TREE")
        return any(marker in command for marker in blocked_markers) or bool(
            GIT_PATH_PATTERN.search(command)
        )

    @staticmethod
    def _touches_git_path(value: Any) -> bool:
        if isinstance(value, dict):
            return any(GradingAgent._touches_git_path(item) for item in value.values())
        if isinstance(value, list):
            return any(GradingAgent._touches_git_path(item) for item in value)
        if not isinstance(value, str):
            return False
        return bool(GIT_PATH_PATTERN.search(value))

    async def run(
        self,
        *,
        cwd: Path,
        system_context: Dict[str, Any],
        user_context: Dict[str, Any],
    ) -> AgentResult:
        """Render prompts, run the agent, and collect the result."""
        options = self._build_options(system_context, cwd)
        prompt = self.renderer.render(PROMPT_DIR, "user", user_context)

        messages: List[Any] = []
        async for message in query(prompt=prompt, options=options):
            messages.append(message)

        return self._collect(messages)

    def _collect(self, messages: List[Any]) -> AgentResult:
        """Extract final text, structured output, and usage from the stream."""
        for message in reversed(messages):
            if isinstance(message, ResultMessage):
                text = getattr(message, "result", "") or ""
                structured = getattr(message, "structured_output", None)
                return AgentResult(
                    text=text,
                    structured=structured,
                    raw_messages=messages,
                    total_cost_usd=getattr(message, "total_cost_usd", None),
                    usage=getattr(message, "usage", None),
                    model_usage=getattr(message, "model_usage", None),
                    num_turns=getattr(message, "num_turns", None),
                    duration_ms=getattr(message, "duration_ms", None),
                    duration_api_ms=getattr(message, "duration_api_ms", None),
                    stop_reason=getattr(message, "stop_reason", None),
                )
        return AgentResult(text="", structured=None, raw_messages=messages)
