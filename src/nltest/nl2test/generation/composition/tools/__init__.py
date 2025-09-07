from typing import List, Tuple

from langchain_core.tools import BaseTool

from .base import BaseCompositionTools
from .grammatical import GrammaticalCompositionTools
from .gherkin import GherkinCompositionTools
from nltest.nl2test.models import DecompositionMode


class CompositionTools:
    """Deprecated: use GrammaticalCompositionTools or GherkinCompositionTools.

    Provided to keep legacy imports working.
    """

    def __init__(
        self,
        *,
        decomposition_mode: DecompositionMode = DecompositionMode.GRAMMATICAL,
        **kwargs,
    ):
        if decomposition_mode == DecompositionMode.GHERKIN:
            self._delegate = GherkinCompositionTools(**kwargs)
        else:
            self._delegate = GrammaticalCompositionTools(**kwargs)

    def all(self) -> Tuple[List[BaseTool], List[BaseTool]]:
        return self._delegate.all()

    def __getattr__(self, item):
        return getattr(self._delegate, item)


__all__ = [
    "BaseCompositionTools",
    "GrammaticalCompositionTools",
    "GherkinCompositionTools",
    "CompositionTools",
]
