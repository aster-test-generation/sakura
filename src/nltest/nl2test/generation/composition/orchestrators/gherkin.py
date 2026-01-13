from __future__ import annotations

from pathlib import Path

from nltest.nl2test.generation.composition.orchestrators.base import (
    BaseCompositionOrchestrator,
)
from cldk.analysis.java import JavaAnalysis
from nltest.nl2test.preprocessing.searchers import MethodSearcher, ClassSearcher
from nltest.nl2test.models import AgentState, LocalizedScenario, NL2TestInput
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.utils.llm import UsageTracker


class GherkinCompositionOrchestrator(BaseCompositionOrchestrator):
    def __init__(
        self,
        *,
        analysis: JavaAnalysis,
        method_searcher: MethodSearcher,
        class_searcher: ClassSearcher,
        nl2_input: NL2TestInput,
        project_root: str,
        test_base_dir: str | Path | None = None,
        module_root: str | Path | None = None,
        usage_tracker: UsageTracker | None = None,
    ) -> None:
        super().__init__(
            analysis=analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            nl2_input=nl2_input,
            project_root=project_root,
            test_base_dir=test_base_dir,
            module_root=module_root,
            decomposition_mode=DecompositionMode.GHERKIN,
            usage_tracker=usage_tracker,
        )

    def assign_task(
        self, blocks, *, instructions: str, agent_state: AgentState | None = None
    ) -> AgentState:
        if self.decomposition_mode != DecompositionMode.GHERKIN:
            raise TypeError("GherkinCompositionOrchestrator is not in GHERKIN mode.")

        localized_scenario: LocalizedScenario = blocks
        if agent_state is not None:
            # Reuse existing state and set current input blocks
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
            localized_scenario=localized_scenario,
        )

        updated_state: AgentState = self.agent.invoke(
            chat_prompt,
            initial_state,
            config={"configurable": {"thread_id": f"cmp:{self.nl2_input.id}"}},
        )
        return updated_state
