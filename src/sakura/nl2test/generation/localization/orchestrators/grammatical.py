from __future__ import annotations

from sakura.nl2test.generation.localization.orchestrators.base import (
    BaseLocalizationOrchestrator,
)
from cldk.analysis.java import JavaAnalysis
from sakura.nl2test.preprocessing.searchers import MethodSearcher, ClassSearcher
from sakura.nl2test.models import (
    AgentState,
    AtomicBlockList,
    NL2TestInput,
)
from sakura.nl2test.models.decomposition import DecompositionMode
from sakura.utils.llm import UsageTracker


class GrammaticalLocalizationOrchestrator(BaseLocalizationOrchestrator):
    def __init__(
        self,
        *,
        analysis: JavaAnalysis,
        method_searcher: MethodSearcher,
        class_searcher: ClassSearcher,
        nl2_input: NL2TestInput,
        usage_tracker: UsageTracker | None = None,
    ) -> None:
        super().__init__(
            analysis=analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            nl2_input=nl2_input,
            decomposition_mode=DecompositionMode.GRAMMATICAL,
            usage_tracker=usage_tracker,
        )

    def assign_task(
        self, blocks, *, instructions: str, agent_state: AgentState | None = None
    ) -> AgentState:
        if self.decomposition_mode != DecompositionMode.GRAMMATICAL:
            raise TypeError(
                "GrammaticalLocalizationOrchestrator is not in GRAMMATICAL mode."
            )

        # Expect an AtomicBlockList as input blocks
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
        prompt_blocks = atomic_blocks.atomic_blocks

        chat_prompt = self.chat_prompt.format(
            nl_description=self.nl2_input.description,
            instructions=instructions,
            blocks=prompt_blocks,
        )

        updated_state: AgentState = self.agent.invoke(
            chat_prompt,
            initial_state,
            config={"configurable": {"thread_id": f"loc:{self.nl2_input.id}"}},
        )
        return updated_state
