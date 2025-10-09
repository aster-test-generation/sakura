from __future__ import annotations

from nltest.nl2test.generation.composition.orchestrators.base import (
    BaseCompositionOrchestrator,
)
from cldk.analysis.java import JavaAnalysis
from nltest.nl2test.preprocessing.searchers import MethodSearcher, ClassSearcher
from nltest.nl2test.models import AgentState, AtomicBlockList, NL2TestInput
from nltest.nl2test.models.decomposition import DecompositionMode


class GrammaticalCompositionOrchestrator(BaseCompositionOrchestrator):
    def __init__(
        self,
        *,
        analysis: JavaAnalysis,
        method_searcher: MethodSearcher,
        class_searcher: ClassSearcher,
        nl2_input: NL2TestInput,
        project_root: str,
    ) -> None:
        super().__init__(
            analysis=analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            nl2_input=nl2_input,
            project_root=project_root,
            decomposition_mode=DecompositionMode.GRAMMATICAL,
        )

    def assign_task(self, blocks, *, instructions: str, agent_state: AgentState | None = None) -> AgentState:
        if self.decomposition_mode != DecompositionMode.GRAMMATICAL:
            raise TypeError(
                "GrammaticalCompositionOrchestrator is not in GRAMMATICAL mode."
            )

        atomic_blocks: AtomicBlockList = blocks
        if agent_state is not None:
            initial_state = (
                agent_state.model_copy(deep=True)
                if hasattr(agent_state, "model_copy")
                else agent_state
            )
            initial_state.atomic_blocks = atomic_blocks
        else:
            initial_state = AgentState(atomic_blocks=atomic_blocks)

        chat_prompt = self.chat_prompt.format(
            nl_description=self.nl2_input.description,
            instructions=instructions,
            atomic_blocks=atomic_blocks.atomic_blocks,
        )

        updated_state: AgentState = self.agent.invoke(
            chat_prompt,
            initial_state,
            config={"configurable": {"thread_id": f"cmp:{self.nl2_input.id}"}},
        )
        return updated_state
