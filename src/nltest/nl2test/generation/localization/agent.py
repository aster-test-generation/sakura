from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.messages import ToolMessage, ToolCall
from langchain_core.tools import BaseTool

from nltest.nl2test.core.react_agent import ReActAgent  # base in this repo
from nltest.nl2test.models import (
    AgentState,
    AtomicBlock,
    AtomicBlockList,
    Scenario,
    LocalizedScenario,
)
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.utils.analysis.common_analysis import CommonAnalysis
from nltest.utils.llm import LLMClient
from nltest.utils.tool_messages import format_tool_ok, format_tool_error


class LocalizationReActAgent(ReActAgent):
    def __init__(
            self,
            *,
            llm: LLMClient,
            tools: List[BaseTool],
            allow_duplicate_tools: List[BaseTool] | None = None,
            system_message: str,
            max_iters: int = 30,
            decomposition_mode: DecompositionMode = DecompositionMode.GRAMMATICAL,
    ):
        super().__init__(
            llm=llm,
            tools=tools,
            allow_duplicate_tools=allow_duplicate_tools,
            system_message=system_message,
            allow_parallelize=True,
            max_iters=max_iters,
        )
        self.decomposition_mode = decomposition_mode

    def prepare_tool_args(
            self, tool_name: str, raw_args: Dict[str, Any], _state: AgentState
    ) -> Tuple[str, Dict[str, Any]]:
        updated_args: Dict[str, Any] = raw_args

        # Normalize class name for CLDK inner classes
        normalize_class = {"get_method_details", "get_reachable_methods_in_class", "get_class_details",
                           "get_inherited_library_classes", "extract_method_code", "get_call_site_details"}
        if tool_name in normalize_class:
            qualified_class_name = raw_args.get("qualified_class_name")
            if isinstance(qualified_class_name, str):
                normalized_class = CommonAnalysis.get_cldk_class_name(qualified_class_name)
                if normalized_class != qualified_class_name:
                    if updated_args is raw_args:
                        updated_args = dict(updated_args)
                    updated_args["qualified_class_name"] = normalized_class

        # Normalize method sig for CLDK constructor methods
        normalize_method_sig = {"get_call_site_details", "get_method_details", "extract_method_code"}
        if tool_name in normalize_method_sig:
            qualified_class_name = updated_args.get("qualified_class_name")
            method_signature = updated_args.get("method_signature")
            normalized_sig = CommonAnalysis.get_cldk_method_sig(
                qualified_class_name, method_signature
            )
            if normalized_sig != method_signature:
                if updated_args is raw_args:
                    updated_args = dict(updated_args)
                updated_args["method_signature"] = normalized_sig

        return tool_name, updated_args

    def process_tool_output(
            self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        tool_name = tool_call["name"]
        handler = {
            "finalize": self.process_finalize_tool_output,
        }.get(tool_name, self.process_generic_tool_output)

        try:
            handler(tool_call, result, state, outputs)
        except Exception as exc:  # pragma: no cover - defensive
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code=type(exc).__name__,
                        message=str(exc),
                        details={
                            "tool": tool_name,
                            "tool_call_id": tool_call["id"],
                        },
                    ),
                    tool_call_id=tool_call["id"],
                )
            )

    def process_finalize_tool_output(
            self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        blocks, comments = result
        state.final_comments = str(comments)
        state.finalize_called = True
        state.force_end_attempts = 0

        if self.decomposition_mode == DecompositionMode.GHERKIN:
            blocks.enforce_candidate_limits()
            state.localized_scenario = blocks  # Expected LocalizedScenario
        else:
            blocks.enforce_candidate_limits()
            state.atomic_blocks = blocks  # Expected AtomicBlockList

        outputs.append(
            ToolMessage(
                content=format_tool_ok({"comments": str(comments)}),
                tool_call_id=tool_call["id"],
            )
        )
        setattr(self, "_end_now", True)

    def process_generic_tool_output(
            self, tool_call: ToolCall, result: Any, _: AgentState, outputs: List
    ) -> None:
        outputs.append(
            ToolMessage(
                content=format_tool_ok(result),
                tool_call_id=tool_call["id"],
            )
        )
