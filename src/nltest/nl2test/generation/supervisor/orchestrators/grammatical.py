from __future__ import annotations

from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.nl2test.models import AgentState, AtomicBlockList

from .base import BaseSupervisorOrchestrator


class GrammaticalSupervisorOrchestrator(BaseSupervisorOrchestrator):
    def __init__(self, **kwargs) -> None:
        super().__init__(decomposition_mode=DecompositionMode.GRAMMATICAL, **kwargs)

    def assign_task(self, blocks):
        if self.decomposition_mode != DecompositionMode.GRAMMATICAL:
            raise TypeError(
                "GrammaticalSupervisorOrchestrator is not in GRAMMATICAL mode."
            )

        atomic_blocks: AtomicBlockList = blocks
        initial_state = AgentState(atomic_blocks=atomic_blocks)

        chat_prompt = self.chat_prompt.format(
            blocks=atomic_blocks,
            nl_description=self.nl2_input.description,
        )

        updated_state: AgentState = self.agent.invoke(chat_prompt, initial_state)
        return updated_state
