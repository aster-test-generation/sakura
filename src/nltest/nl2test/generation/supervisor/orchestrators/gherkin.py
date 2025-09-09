from __future__ import annotations

from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.nl2test.models import AgentState, LocalizedScenario

from .base import BaseSupervisorOrchestrator


class GherkinSupervisorOrchestrator(BaseSupervisorOrchestrator):
    def __init__(self, **kwargs) -> None:
        super().__init__(decomposition_mode=DecompositionMode.GHERKIN, **kwargs)

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
