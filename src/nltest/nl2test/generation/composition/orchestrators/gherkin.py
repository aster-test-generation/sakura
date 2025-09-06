from __future__ import annotations

from nltest.nl2test.generation.composition.orchestrators.base import (
    BaseCompositionOrchestrator,
)
from cldk.analysis.java import JavaAnalysis
from nltest.nl2test.preprocessing.searchers import MethodSearcher, ClassSearcher
from nltest.nl2test.models import AgentState, LocalizedScenario, NL2TestInput
from nltest.nl2test.models.decomposition import DecompositionMode


class GherkinCompositionOrchestrator(BaseCompositionOrchestrator):
    def __init__(
        self,
        *,
        analysis: JavaAnalysis,
        method_searcher: MethodSearcher,
        class_searcher: ClassSearcher,
        nl2_input: NL2TestInput,
        base_project_dir: str | None = None,
    ) -> None:
        super().__init__(
            analysis=analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            nl2_input=nl2_input,
            base_project_dir=base_project_dir,
            decomposition_mode=DecompositionMode.GHERKIN,
        )

    def assign_task(self, blocks, *, instructions: str):
        if self.decomposition_mode != DecompositionMode.GHERKIN:
            raise TypeError("GherkinCompositionOrchestrator is not in GHERKIN mode.")

        localized_scenario: LocalizedScenario = blocks
        initial_state = AgentState(localized_scenario=localized_scenario)

        chat_prompt = self.chat_prompt.format(
            nl_description=self.nl2_input.description,
            instructions=instructions,
            localized_scenario=localized_scenario,
        )

        updated_state: AgentState = self.agent.invoke(chat_prompt, initial_state)
        return updated_state.localized_scenario, (
            updated_state.final_comments or "No comments."
        )
