from __future__ import annotations

from .base import BaseCompositionTools


class GherkinCompositionTools(BaseCompositionTools):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Placeholder for gherkin-mode-specific tools; extend here when needed

