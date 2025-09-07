from __future__ import annotations

from typing import Tuple

from nltest.nl2test.generation.localization.orchestrators.base import (
    BaseLocalizationOrchestrator,
)
from cldk.analysis.java import JavaAnalysis
from nltest.nl2test.preprocessing.searchers import MethodSearcher, ClassSearcher
from nltest.nl2test.models import (
    AgentState,
    AtomicBlockList,
    NL2TestInput,
)
from nltest.nl2test.models.decomposition import DecompositionMode


class GrammaticalLocalizationOrchestrator(BaseLocalizationOrchestrator):
    def __init__(
        self,
        *,
        analysis: JavaAnalysis,
        method_searcher: MethodSearcher,
        class_searcher: ClassSearcher,
        nl2_input: NL2TestInput,
    ) -> None:
        super().__init__(
            analysis=analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            nl2_input=nl2_input,
            decomposition_mode=DecompositionMode.GRAMMATICAL,
        )

    def assign_task(self, blocks, *, instructions: str):
        if self.decomposition_mode != DecompositionMode.GRAMMATICAL:
            raise TypeError(
                "GrammaticalLocalizationOrchestrator is not in GRAMMATICAL mode."
            )

        # Expect an AtomicBlockList as input blocks
        atomic_blocks: AtomicBlockList = blocks
        initial_state = AgentState(atomic_blocks=atomic_blocks)
        prompt_blocks = atomic_blocks.atomic_blocks

        chat_prompt = self.chat_prompt.format(
            nl_description=self.nl2_input.description,
            instructions=instructions,
            blocks=prompt_blocks,
        )

        updated_state: AgentState = self.agent.invoke(chat_prompt, initial_state)
        if not updated_state.atomic_blocks:
            raise ValueError("Agent did not return AtomicBlockList in GRAMMATICAL mode")
        return updated_state.atomic_blocks, (updated_state.final_comments or "No comments.")
