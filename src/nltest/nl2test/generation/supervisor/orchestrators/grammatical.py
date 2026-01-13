from __future__ import annotations
from pathlib import Path
from typing import Tuple

from cldk.analysis.java import JavaAnalysis
from nltest.nl2test.preprocessing.searchers import MethodSearcher, ClassSearcher
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.nl2test.models import AgentState, AtomicBlockList, NL2TestInput
from nltest.utils.llm import UsageTracker

from .base import BaseSupervisorOrchestrator


class GrammaticalSupervisorOrchestrator(BaseSupervisorOrchestrator):
    def __init__(
        self,
        *,
        analysis: JavaAnalysis,
        method_searcher: MethodSearcher,
        class_searcher: ClassSearcher,
        nl2_input: NL2TestInput,
        base_project_dir: str,
        test_base_dir: str | Path | None = None,
        module_root: str | Path | None = None,
        usage_tracker: UsageTracker | None = None,
    ) -> None:
        super().__init__(
            analysis=analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            nl2_input=nl2_input,
            decomposition_mode=DecompositionMode.GRAMMATICAL,
            base_project_dir=base_project_dir,
            test_base_dir=test_base_dir,
            module_root=module_root,
            usage_tracker=usage_tracker,
        )

    def assign_task(self, blocks) -> Tuple[AgentState, AgentState, AgentState]:
        if self.decomposition_mode != DecompositionMode.GRAMMATICAL:
            raise TypeError(
                "GrammaticalSupervisorOrchestrator is not in GRAMMATICAL mode."
            )

        atomic_blocks: AtomicBlockList = blocks
        initial_state = AgentState(atomic_blocks=atomic_blocks)

        chat_prompt = self.chat_prompt.format(
            blocks=atomic_blocks,
            nl_description=self.nl2_input.description,
        )

        updated_state: AgentState = self.agent.invoke(chat_prompt, initial_state)
        if updated_state.curr_tool_trajectory:
            updated_state.tool_trajectories.append(
                updated_state.curr_tool_trajectory.copy()
            )
            updated_state.curr_tool_trajectory.clear()
        localization_state: AgentState = self.agent.localization_state
        composition_state: AgentState = self.agent.composition_state
        return updated_state, localization_state, composition_state
