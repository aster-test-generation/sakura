from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UsageTracker:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def record(self, input_tokens: int, output_tokens: int) -> None:
        """Accumulate token usage for a single LLM invocation."""
        self.calls += 1
        self.input_tokens += input_tokens or 0
        self.output_tokens += output_tokens or 0

    def reset(self) -> None:
        """Clear all tracked counts."""
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def totals(self) -> dict[str, int]:
        """Return a snapshot of the tracked counts."""
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
