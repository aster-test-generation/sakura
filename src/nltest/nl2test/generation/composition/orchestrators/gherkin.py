from __future__ import annotations

from nltest.nl2test.generation.composition.orchestrators.base import (
    BaseCompositionOrchestrator,
)
from nltest.nl2test.models import AgentState, LocalizedScenario
from nltest.nl2test.models.decomposition import DecompositionMode


class GherkinCompositionOrchestrator(BaseCompositionOrchestrator):
    def __init__(self, **kwargs) -> None:
        super().__init__(decomposition_mode=DecompositionMode.GHERKIN, **kwargs)

    def assign_task(self, blocks, *, instructions: str):
        if self.decomposition_mode != DecompositionMode.GHERKIN:
            raise TypeError("GherkinCompositionOrchestrator is not in GHERKIN mode.")

        localized_scenario: LocalizedScenario = blocks
        initial_state = AgentState(localized_scenario=localized_scenario)

        # TODO: Revise composition prompt
        chat_prompt = self.chat_prompt.format(
            nl_description=self.nl2_input.description,
            instructions=instructions,
            atomic_blocks=localized_scenario,
        )

        updated_state: AgentState = self.agent.invoke(chat_prompt, initial_state)
        return updated_state.localized_scenario, (
            updated_state.final_comments or "No comments."
        )
