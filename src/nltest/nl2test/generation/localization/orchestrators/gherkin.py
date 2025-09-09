from __future__ import annotations

from typing import Tuple

from nltest.nl2test.generation.localization.orchestrators.base import (
    BaseLocalizationOrchestrator,
)
from nltest.nl2test.models import AgentState, LocalizedScenario
from nltest.nl2test.models.decomposition import DecompositionMode


class GherkinLocalizationOrchestrator(BaseLocalizationOrchestrator):
    def __init__(self, **kwargs) -> None:
        super().__init__(decomposition_mode=DecompositionMode.GHERKIN, **kwargs)

    def assign_task(self, blocks, *, instructions: str, agent_state: AgentState | None = None) -> AgentState:
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

        updated_state: AgentState = self.agent.invoke(chat_prompt, initial_state)
        return updated_state
