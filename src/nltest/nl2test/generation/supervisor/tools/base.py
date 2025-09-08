from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

from langchain_core.tools import BaseTool

from nltest.nl2test.core.react_agent import ReActAgent
from nltest.nl2test.models import NL2TestInput
from nltest.utils.llm.llm_client import LLMClient


class BaseSupervisorTools:
    """Minimal supervisor tools base.

    Provides the same surface as other tool bases: a `tools` list,
    `allow_duplicate_tools`, and an `all()` accessor. Mode-specific
    subclasses can append tools later.
    """

    def __init__(
        self,
        *,
        llm: Optional[LLMClient] = None,
        project_root: Union[str, Path, None] = None,
        nl2_input: Optional[NL2TestInput] = None,
        localization_agent: Optional[ReActAgent] = None,
        composition_agent: Optional[ReActAgent] = None,
    ) -> None:
        self.llm = llm
        self.project_root = Path(project_root) if project_root is not None else None
        self.nl2_input = nl2_input
        self.localization_agent = localization_agent
        self.composition_agent = composition_agent

        self.tools: List[BaseTool] = []
        self.allow_duplicate_tools: List[BaseTool] = []

    def all(self) -> Tuple[List[BaseTool], List[BaseTool]]:
        return self.tools, self.allow_duplicate_tools

