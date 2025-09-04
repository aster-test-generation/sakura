from __future__ import annotations

from nltest.nl2test.generation.composition.orchestrators.base import (
    BaseCompositionOrchestrator,
)
from nltest.nl2test.models import AgentState, AtomicBlockList
from nltest.nl2test.models.decomposition import DecompositionMode


class GrammaticalCompositionOrchestrator(BaseCompositionOrchestrator):
    def __init__(self, **kwargs) -> None:
        super().__init__(decomposition_mode=DecompositionMode.GRAMMATICAL, **kwargs)

    def assign_task(self, blocks, *, instructions: str):
        if self.decomposition_mode != DecompositionMode.GRAMMATICAL:
            raise TypeError(
                "GrammaticalCompositionOrchestrator is not in GRAMMATICAL mode."
            )

        atomic_blocks: AtomicBlockList = blocks
        initial_state = AgentState(atomic_blocks=atomic_blocks)

        chat_prompt = self.chat_prompt.format(
            nl_description=self.nl2_input.description,
            instructions=instructions,
            atomic_blocks=atomic_blocks.atomic_blocks,
        )

        updated_state: AgentState = self.agent.invoke(chat_prompt, initial_state)
        return updated_state.atomic_blocks, (
            updated_state.final_comments or "No comments."
        )
