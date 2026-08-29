from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from sakura.nl2test.core.deferred_tool import DeferredTool
from sakura.nl2test.core.react_agent import ReActAgent
from sakura.nl2test.models import AgentState
from sakura.utils.llm import LLMClient


class _OfflineFinalizingReActAgent(ReActAgent):
    def _execute_force_end(self, state: AgentState) -> AgentState:
        state.force_end_attempts += 1
        state.finalize_called = True
        state.final_comments = "Finalized at the iteration limit."
        return state


def test_max_iters_is_hard_limit_for_prose_only_llm() -> None:
    llm = MagicMock(spec=LLMClient)
    llm.invoke_messages.return_value = AIMessage(content="Prose without a tool call.")
    finalize_tool = DeferredTool.create_no_args(
        name="finalize",
        description="Finalize the agent.",
    )
    agent = _OfflineFinalizingReActAgent(
        llm=llm,
        tools=[finalize_tool],
        max_iters=1,
        use_checkpointer=False,
    )

    result = agent.invoke("Complete the task.")

    assert result.finalize_called is True
    assert result.iterations == 1
    llm.invoke_messages.assert_called_once()
