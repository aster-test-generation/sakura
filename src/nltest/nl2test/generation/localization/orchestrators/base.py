from __future__ import annotations

from typing import List, Tuple

from cldk.analysis.java import JavaAnalysis
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from nltest.nl2test.generation.localization.agent import LocalizationReActAgent
from nltest.nl2test.generation.localization.tools import (
    GherkinLocalizationTools,
    GrammaticalLocalizationTools,
)
from nltest.nl2test.models import AgentState, NL2TestInput
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.nl2test.preprocessing.searchers import ClassSearcher, MethodSearcher
from nltest.utils.config import Config
from nltest.utils.llm.llm_client import ClientType, LLMClient


class BaseLocalizationOrchestrator:
    def __init__(
        self,
        *,
        analysis: JavaAnalysis,
        method_searcher: MethodSearcher,
        class_searcher: ClassSearcher,
        nl2_input: NL2TestInput,
        decomposition_mode: DecompositionMode,
    ) -> None:
        decision_llm = LLMClient(ClientType.DECISION)

        if decomposition_mode == DecompositionMode.GHERKIN:
            tool_builder = GherkinLocalizationTools(
                analysis=analysis,
                method_searcher=method_searcher,
                class_searcher=class_searcher,
            )
        else:
            tool_builder = GrammaticalLocalizationTools(
                analysis=analysis,
                method_searcher=method_searcher,
                class_searcher=class_searcher,
            )

        tools, allow_duplicate_tools = tool_builder.all()

        self.nl2_input = nl2_input
        self.decomposition_mode = decomposition_mode

        chat_prompt, system_prompt = self._init_prompts()
        self.chat_prompt = chat_prompt

        # Determine if the model supports parallel tool calls and pass max_iters to the system prompt
        parallelizable: bool = decision_llm.can_parallel_tool_call()
        max_iters = Config().get("localization", "max_iters")
        system_message = system_prompt.format(
            parallelizable=parallelizable,
            max_iters=max_iters,
        )

        self.agent = LocalizationReActAgent(
            llm=decision_llm,
            tools=tools,
            allow_duplicate_tools=allow_duplicate_tools,
            max_iters=max_iters,
            system_message=system_message,
            decomposition_mode=decomposition_mode,
        )

    def _init_prompts(self) -> tuple[PromptTemplate, PromptTemplate]:
        if self.decomposition_mode == DecompositionMode.GHERKIN:
            chat_file = "localization_agent_gherkin.jinja2"
            system_file = "localization_agent_gherkin.jinja2"
        else:
            chat_file = "localization_agent_grammatical.jinja2"
            system_file = "localization_agent_grammatical.jinja2"

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
