from __future__ import annotations
from pathlib import Path
from typing import Tuple

from cldk.analysis.java import JavaAnalysis
from nltest.nl2test.preprocessing.searchers import MethodSearcher, ClassSearcher
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.nl2test.models import AgentState, LocalizedScenario, NL2TestInput
from nltest.utils.llm import UsageTracker

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
        test_base_dir: str | Path | None = None,
        usage_tracker: UsageTracker | None = None,
    ) -> None:
        super().__init__(
            analysis=analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            nl2_input=nl2_input,
            decomposition_mode=DecompositionMode.GHERKIN,
            base_project_dir=base_project_dir,
            test_base_dir=test_base_dir,
            usage_tracker=usage_tracker,
        )

    def assign_task(self, blocks) -> Tuple[AgentState, AgentState, AgentState]:
        if self.decomposition_mode != DecompositionMode.GHERKIN:
            raise TypeError("GherkinSupervisorOrchestrator is not in GHERKIN mode.")

        localized_scenario: LocalizedScenario = blocks
        initial_state = AgentState(localized_scenario=localized_scenario)

        chat_prompt = self.chat_prompt.format(
            blocks=localized_scenario,
            nl_description=self.nl2_input.description,
        )

        supervisor_state: AgentState = self.agent.invoke(
            chat_prompt,
            initial_state,
            config={"configurable": {"thread_id": f"sup:{self.nl2_input.id}"}},
        )
        if supervisor_state.curr_tool_trajectory:
            supervisor_state.tool_trajectories.append(
                supervisor_state.curr_tool_trajectory.copy()
            )
            supervisor_state.curr_tool_trajectory.clear()
        localization_state: AgentState = self.agent.localization_state
        composition_state: AgentState = self.agent.composition_state
        return supervisor_state, localization_state, composition_state
