from __future__ import annotations

from cldk.analysis.java import JavaAnalysis
from nltest.nl2test.preprocessing.searchers import MethodSearcher, ClassSearcher
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.nl2test.models import AgentState, AtomicBlockList, NL2TestInput

from .base import BaseSupervisorOrchestrator


class GrammaticalSupervisorOrchestrator(BaseSupervisorOrchestrator):
    def __init__(
        self,
        *,
        analysis: JavaAnalysis,
        method_searcher: MethodSearcher,
        class_searcher: ClassSearcher,
        nl2_input: NL2TestInput,
        base_project_dir: str,
    ) -> None:
        super().__init__(
            analysis=analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            nl2_input=nl2_input,
            decomposition_mode=DecompositionMode.GRAMMATICAL,
            base_project_dir=base_project_dir,
        )

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
