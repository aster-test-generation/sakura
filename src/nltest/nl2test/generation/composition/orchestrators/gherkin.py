from __future__ import annotations

from nltest.nl2test.generation.composition.orchestrators.base import (
    BaseCompositionOrchestrator,
)
from nltest.nl2test.models.decomposition import DecompositionMode


class GherkinCompositionOrchestrator(BaseCompositionOrchestrator):
    def __init__(self, **kwargs) -> None:
        super().__init__(decomposition_mode=DecompositionMode.GHERKIN, **kwargs)

    def assign_task(self, blocks, *, instructions: str):
        # Composition for GHERKIN mode is not yet implemented. This scaffold mirrors localization.
        raise NotImplementedError(
            "GherkinCompositionOrchestrator.assign_task is not implemented yet."
        )

