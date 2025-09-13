from __future__ import annotations

from cldk.analysis.java import JavaAnalysis
from nltest.nl2test.preprocessing.searchers import MethodSearcher, ClassSearcher
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.nl2test.models import AgentState, LocalizedScenario, NL2TestInput

from .base import BaseSupervisorOrchestrator


class GherkinSupervisorOrchestrator(BaseSupervisorOrchestrator):
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
            decomposition_mode=DecompositionMode.GHERKIN,
            base_project_dir=base_project_dir,
        )

    def assign_task(self, blocks):
        if self.decomposition_mode != DecompositionMode.GHERKIN:
            raise TypeError("GherkinSupervisorOrchestrator is not in GHERKIN mode.")

        localized_scenario: LocalizedScenario = blocks
        initial_state = AgentState(localized_scenario=localized_scenario)

        chat_prompt = self.chat_prompt.format(
            blocks=localized_scenario,
            nl_description=self.nl2_input.description,
        )

        updated_state: AgentState = self.agent.invoke(chat_prompt, initial_state)
        return updated_state
