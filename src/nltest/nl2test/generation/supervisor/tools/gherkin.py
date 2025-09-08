from __future__ import annotations

from .base import BaseSupervisorTools


class GherkinSupervisorTools(BaseSupervisorTools):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

