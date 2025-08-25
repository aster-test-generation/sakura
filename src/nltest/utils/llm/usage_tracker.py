from __future__ import annotations
from dataclasses import dataclass, asdict
from threading import Lock
from typing import Dict, Optional

@dataclass
class Totals:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

class UsageTracker:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._lock = Lock()
                cls._instance._totals = Totals()
                cls._instance._baseline = None
            return cls._instance

    def record(self, model: str, input_tokens: int, output_tokens: int, total_tokens: Optional[int] = None) -> None:
        if total_tokens is None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)

        with self._lock:
            # global
            totals = self._totals
            totals.calls += 1
            totals.input_tokens += input_tokens or 0
            totals.output_tokens += output_tokens or 0
            totals.total_tokens += total_tokens or 0

    def snapshot(self) -> Dict[str, int]:
        """Return current input and output tokens overall."""
        with self._lock:
            return {
                "input_tokens": self._totals.input_tokens,
                "output_tokens": self._totals.output_tokens,
            }

    def start(self) -> None:
        """Store current snapshot as baseline for delta tracking."""
        with self._lock:
            self._baseline = {
                "input_tokens": self._totals.input_tokens,
                "output_tokens": self._totals.output_tokens,
            }

    def stop(self) -> Dict[str, int]:
        """Return delta of input and output tokens since start() was called."""
        with self._lock:
            if self._baseline is None:
                return {"input_tokens": 0, "output_tokens": 0}
            
            current = {
                "input_tokens": self._totals.input_tokens,
                "output_tokens": self._totals.output_tokens,
            }
            return {
                "input_tokens": current["input_tokens"] - self._baseline["input_tokens"],
                "output_tokens": current["output_tokens"] - self._baseline["output_tokens"],
            }

    def reset(self) -> None:
        with self._lock:
            self._totals = Totals()
            self._baseline = None

usage_tracker = UsageTracker()
