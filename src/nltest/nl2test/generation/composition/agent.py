from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

from langchain_core.messages import ToolMessage, ToolCall
from langchain_core.tools import BaseTool

from nltest.nl2test.core.react_agent import ReActAgent
from nltest.nl2test.models import AgentState, DecompositionMode
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.utils.llm.llm_client import LLMClient


class CompositionReActAgent(ReActAgent):
    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: List[BaseTool],
        system_message: str,
        max_iters: int = 8,
        decomposition_mode: DecompositionMode = DecompositionMode.GRAMMATICAL,
    ):
        # Allow caller to inject system_message for parity with localization

        super().__init__(
            llm=llm, tools=tools, system_message=system_message, max_iters=max_iters
        )

    def _prepare_tool_args(
        self, tool_name: str, raw_args: Dict, state: AgentState
    ) -> Tuple[str, Dict]:
        """Implementation for preparing tool call arguments."""
        return tool_name, raw_args

    def _process_tool_output(
        self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        """Process tool output and mutate state when appropriate."""

        if tool_call["name"] == "modify_scenario_comment":
            # Tool returns args directly
            step_id: Optional[int] = None
            comment: Optional[str] = None
            order: Optional[int] = None
            note: Optional[str] = None

            if isinstance(result, (list, tuple)) and len(result) == 2:
                # Disambiguate by which state is present
                if state.atomic_blocks is not None:
                    order, note = int(result[0]), str(result[1])
                else:
                    step_id, comment = int(result[0]), str(result[1])
            elif isinstance(result, dict):
                step_id = result.get("id")
                comment = result.get("comment")
                order = result.get("order")
                note = result.get("note")

            # GRAMMATICAL mode (flattened atomic blocks by order)
            if state.atomic_blocks is not None:
                if order is None or note is None:
                    outputs.append(
                        ToolMessage(
                            content="Invalid modify_scenario_comment args for GRAMMATICAL mode.",
                            tool_call_id=tool_call["id"],
                        )
                    )
                    return

                found = False
                for b in state.atomic_blocks.atomic_blocks:
                    if b.order == int(order):
                        b.notes = str(note)
                        found = True
                        break

                if found:
                    outputs.append(
                        ToolMessage(
                            content=f"Updated note for block order {order}.",
                            tool_call_id=tool_call["id"],
                        )
                    )
                else:
                    outputs.append(
                        ToolMessage(
                            content=f"No atomic block with order {order} exists.",
                            tool_call_id=tool_call["id"],
                        )
                    )
                return

            # GHERKIN mode (LocalizedScenario with LocalizedSteps)
            if state.localized_scenario is not None:
                if step_id is None or comment is None:
                    outputs.append(
                        ToolMessage(
                            content="Invalid modify_scenario_comment args for GHERKIN mode.",
                            tool_call_id=tool_call["id"],
                        )
                    )
                    return

                found = False
                # Search setup
                for s in state.localized_scenario.setup:
                    if s.id == int(step_id):
                        s.comments = str(comment)
                        found = True
                        break

                # Search steps: given/when/then
                if not found:
                    for gstep in state.localized_scenario.steps:
                        for group in (gstep.given, gstep.when, gstep.then):
                            for s in group:
                                if s.id == int(step_id):
                                    s.comments = str(comment)
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break

                # Search teardown
                if not found:
                    for s in state.localized_scenario.teardown:
                        if s.id == int(step_id):
                            s.comments = str(comment)
                            found = True
                            break

                if found:
                    outputs.append(
                        ToolMessage(
                            content=f"Updated comment for step id {step_id}.",
                            tool_call_id=tool_call["id"],
                        )
                    )
                else:
                    outputs.append(
                        ToolMessage(
                            content=f"No step with id {step_id} exists.",
                            tool_call_id=tool_call["id"],
                        )
                    )
                return

            # No recognizable state
            outputs.append(
                ToolMessage(
                    content="No scenario or atomic blocks present to modify.",
                    tool_call_id=tool_call["id"],
                )
            )
            return

        # Default: echo tool result
        outputs.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )
