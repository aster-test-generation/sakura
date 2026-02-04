from __future__ import annotations

from sakura.nl2test.generation.localization.orchestrators.base import (
    BaseLocalizationOrchestrator,
)
from cldk.analysis.java import JavaAnalysis
from sakura.nl2test.preprocessing.searchers import MethodSearcher, ClassSearcher
from sakura.nl2test.models import AgentState, LocalizedScenario, NL2TestInput
from sakura.utils.llm import UsageTracker
from sakura.nl2test.models.decomposition import DecompositionMode


class GherkinLocalizationOrchestrator(BaseLocalizationOrchestrator):
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
            decomposition_mode=DecompositionMode.GHERKIN,
            usage_tracker=usage_tracker,
        )

    def assign_task(
        self, blocks, *, instructions: str, agent_state: AgentState | None = None
    ) -> AgentState:
        if self.decomposition_mode != DecompositionMode.GHERKIN:
            raise TypeError("GherkinLocalizationOrchestrator is not in GHERKIN mode.")

        # Expect a LocalizedScenario as input blocks
        localized_scenario: LocalizedScenario = blocks
        if agent_state is not None:
            initial_state = (
                agent_state.model_copy(deep=True)
                if hasattr(agent_state, "model_copy")
                else agent_state
            )
            initial_state.localized_scenario = localized_scenario
        else:
            initial_state = AgentState(localized_scenario=localized_scenario)

        chat_prompt = self.chat_prompt.format(
            nl_description=self.nl2_input.description,
            instructions=instructions,
            steps=localized_scenario,
        )

        updated_state: AgentState = self.agent.invoke(
            chat_prompt,
            initial_state,
            config={"configurable": {"thread_id": f"loc:{self.nl2_input.id}"}},
        )
        return updated_state
