from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseDecomposer(ABC):
    @abstractmethod
    def decompose(self, nl_description: str) -> Any:
        raise NotImplementedError

