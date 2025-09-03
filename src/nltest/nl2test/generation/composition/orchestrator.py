from __future__ import annotations

from typing import Tuple

from cldk.analysis.java import JavaAnalysis

from nltest.nl2test.generation.composition.orchestrators import (
    GrammaticalCompositionOrchestrator,
    GherkinCompositionOrchestrator,
)
from nltest.nl2test.models import AtomicBlockList, NL2TestInput
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.nl2test.preprocessing.searchers import ClassSearcher, MethodSearcher


class CompositionOrchestrator:
    """Compatibility wrapper that delegates to mode-specific orchestrators.

    Default mode is GRAMMATICAL to match legacy usage.
    """

    def __init__(
        self,
        analysis: JavaAnalysis,
        method_searcher: MethodSearcher,
        class_searcher: ClassSearcher,
        nl2_input: NL2TestInput,
        *,
        decomposition_mode: DecompositionMode = DecompositionMode.GRAMMATICAL,
        base_project_dir: str | None = None,
    ) -> None:
        if decomposition_mode == DecompositionMode.GHERKIN:
            self._delegate = GherkinCompositionOrchestrator(
                analysis=analysis,
                method_searcher=method_searcher,
                class_searcher=class_searcher,
                nl2_input=nl2_input,
                base_project_dir=base_project_dir,
            )
        else:
            self._delegate = GrammaticalCompositionOrchestrator(
                analysis=analysis,
                method_searcher=method_searcher,
                class_searcher=class_searcher,
                nl2_input=nl2_input,
                base_project_dir=base_project_dir,
            )

    # Backwards-compatible signature (grammatical blocks)
    def assign_task(self, instructions: str, atomic_blocks: AtomicBlockList) -> Tuple[AtomicBlockList, str]:
        return self._delegate.assign_task(atomic_blocks, instructions=instructions)
