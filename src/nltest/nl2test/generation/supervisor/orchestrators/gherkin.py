from __future__ import annotations

from nltest.nl2test.models.decomposition import DecompositionMode

from .base import BaseSupervisorOrchestrator


class GherkinSupervisorOrchestrator(BaseSupervisorOrchestrator):
    def __init__(self, **kwargs) -> None:
        super().__init__(decomposition_mode=DecompositionMode.GHERKIN, **kwargs)

