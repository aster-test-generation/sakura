"""Jinja2 renderer for the agent-grading prompt templates.

Ported from sakura-uncertainty (src/sakura_uncertainty/agents/prompt_renderer.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPTS_DIR = Path(__file__).parent / "prompts"


class PromptRenderer:
    """Renders Jinja2 prompt templates bundled with this package."""

    def __init__(self, prompts_dir: Path = PROMPTS_DIR) -> None:
        self.prompts_dir = prompts_dir
        self._env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    def render(self, prompt_dir: str, name: str, context: Dict[str, Any]) -> str:
        """Render ``<prompt_dir>/<name>.j2`` with the given context.

        Missing variables raise (StrictUndefined).
        """
        template = self._env.get_template(f"{prompt_dir}/{name}.j2")
        return template.render(**context)
