from __future__ import annotations

from typing import List

from cldk.analysis.java import JavaAnalysis
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from nltest.nl2test.generation.composition.agent import CompositionReActAgent
from nltest.nl2test.generation.composition.tools import (
    GherkinCompositionTools,
    GrammaticalCompositionTools,
)
from nltest.nl2test.models import AgentState, NL2TestInput
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.nl2test.preprocessing.searchers import ClassSearcher, MethodSearcher
from nltest.utils.config import Config
from nltest.utils.llm.llm_client import ClientType, LLMClient


class BaseCompositionOrchestrator:
    def __init__(
        self,
        *,
        analysis: JavaAnalysis,
        method_searcher: MethodSearcher,
        class_searcher: ClassSearcher,
        nl2_input: NL2TestInput,
        base_project_dir: str | None = None,
        decomposition_mode: DecompositionMode,
    ) -> None:
        decision_llm = LLMClient(ClientType.DECISION)
        structured_llm = LLMClient(ClientType.STRUCTURED)

        tool_builder = (
            GherkinCompositionTools(
                analysis=analysis,
                method_searcher=method_searcher,
                class_searcher=class_searcher,
                structured_llm=structured_llm,
                base_project_dir=base_project_dir or ".",
                nl2_input=nl2_input,
            )
            if decomposition_mode == DecompositionMode.GHERKIN
            else GrammaticalCompositionTools(
                analysis=analysis,
                method_searcher=method_searcher,
                class_searcher=class_searcher,
                structured_llm=structured_llm,
                base_project_dir=base_project_dir or ".",
                nl2_input=nl2_input,
            )
        )

        tools: List[BaseTool] = tool_builder.all()

        self.nl2_input = nl2_input
        self.decomposition_mode = decomposition_mode

        chat_prompt, system_prompt = self._init_prompts()
        self.chat_prompt = chat_prompt
        system_message = system_prompt.format()

        self.agent = CompositionReActAgent(
            llm=decision_llm,
            tools=tools,
            max_iters=Config().get("composition", "max_iters"),
            system_message=system_message,
        )

    def _init_prompts(self) -> tuple[PromptTemplate, PromptTemplate]:
        # For now, composition uses a single prompt template for both chat/system
        chat_file = "composition_agent.jinja2"
        system_file = "composition_agent.jinja2"

        chat_prompt = LoadPrompt.load_prompt(
            chat_file, PromptFormat.JINJA2, prompt_type="chat"
        )
        system_prompt = LoadPrompt.load_prompt(
            system_file, PromptFormat.JINJA2, prompt_type="system"
        )
        return chat_prompt, system_prompt

    # Shared signature implemented by subclasses. Intentionally untyped for blocks/output.
    def assign_task(self, blocks, *, instructions: str):  # pragma: no cover - interface
        raise NotImplementedError
