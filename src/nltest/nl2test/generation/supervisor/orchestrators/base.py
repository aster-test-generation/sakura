from __future__ import annotations

from typing import Any, Optional

from cldk.analysis.java import JavaAnalysis

from nltest.nl2test.models import NL2TestInput
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.nl2test.preprocessing.searchers import ClassSearcher, MethodSearcher


class BaseSupervisorOrchestrator:
    """Minimal supervisor orchestrator base.

    Mirrors the structure of composition/localization orchestrator bases,
    but keeps only constructor wiring until specific behavior is added.
    """

    def __init__(
        self,
        *,
        analysis: Optional[JavaAnalysis] = None,
        method_searcher: Optional[MethodSearcher] = None,
        class_searcher: Optional[ClassSearcher] = None,
        nl2_input: Optional[NL2TestInput] = None,
        decomposition_mode: DecompositionMode = DecompositionMode.GRAMMATICAL,
        base_project_dir: str | None = None,
    ) -> None:
        self.analysis = analysis
        self.method_searcher = method_searcher
        self.class_searcher = class_searcher
        self.nl2_input = nl2_input
        self.decomposition_mode = decomposition_mode
        self.base_project_dir = base_project_dir

    # Intentionally untyped to allow different block types per mode
    def assign_task(self, *args: Any, **kwargs: Any):  # pragma: no cover - interface
        raise NotImplementedError

