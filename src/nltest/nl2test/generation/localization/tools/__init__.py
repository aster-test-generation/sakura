from typing import List

from langchain_core.tools import BaseTool

from .base import BaseLocalizationTools
from .grammatical import GrammaticalLocalizationTools
from .gherkin import GherkinLocalizationTools
from nltest.nl2test.models import DecompositionMode


class LocalizationTools:
    """Deprecated: use GrammaticalLocalizationTools or GherkinLocalizationTools.

    Provided to keep tests and legacy imports working.
    """

    def __init__(
        self,
        *,
        decomposition_mode: DecompositionMode = DecompositionMode.GRAMMATICAL,
        **kwargs,
    ):
        if decomposition_mode == DecompositionMode.GHERKIN:
            self._delegate = GherkinLocalizationTools(**kwargs)
        else:
            self._delegate = GrammaticalLocalizationTools(**kwargs)

    def all(self) -> List[BaseTool]:
        return self._delegate.all()

    def __getattr__(self, item):
        # Forward any attribute/method access to the delegate for compatibility
        return getattr(self._delegate, item)


__all__ = [
    "BaseLocalizationTools",
    "GrammaticalLocalizationTools",
    "GherkinLocalizationTools",
    "LocalizationTools",
]
