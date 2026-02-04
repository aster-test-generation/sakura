from __future__ import annotations

from pathlib import Path
from typing import List

from cldk.analysis.java import JavaAnalysis
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from sakura.nl2test.generation.composition.agent import CompositionReActAgent
from sakura.nl2test.generation.composition.tools import (
    GherkinCompositionTools,
    GrammaticalCompositionTools,
)
from sakura.nl2test.models import AgentState, NL2TestInput
from sakura.nl2test.models.decomposition import DecompositionMode
from sakura.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from sakura.nl2test.preprocessing.searchers import ClassSearcher, MethodSearcher
from sakura.utils.config import Config
from sakura.utils.llm import ClientType, LLMClient, UsageTracker


class BaseCompositionOrchestrator:
    def __init__(
        self,
        *,
        analysis: JavaAnalysis,
        method_searcher: MethodSearcher,
        class_searcher: ClassSearcher,
        nl2_input: NL2TestInput,
        project_root: str,
        test_base_dir: str | Path | None = None,
        module_root: str | Path | None = None,
        decomposition_mode: DecompositionMode,
        usage_tracker: UsageTracker | None = None,
    ) -> None:
        self.usage_tracker = usage_tracker or UsageTracker()
        decision_llm = LLMClient(
            ClientType.DECISION,
            usage_tracker=self.usage_tracker,
        )
        structured_llm = LLMClient(
            ClientType.STRUCTURED,
            usage_tracker=self.usage_tracker,
        )

        resolved_project_root = Path(project_root).expanduser().resolve()
        resolved_module_root = resolved_project_root
        if module_root is not None:
            resolved_module_root = Path(module_root).expanduser()
            if not resolved_module_root.is_absolute():
                resolved_module_root = resolved_project_root / resolved_module_root
            resolved_module_root = resolved_module_root.resolve()

        tool_builder = (
            GherkinCompositionTools(
                analysis=analysis,
                method_searcher=method_searcher,
                class_searcher=class_searcher,
                structured_llm=structured_llm,
                project_root=str(resolved_project_root),
                module_root=resolved_module_root,
                nl2_input=nl2_input,
            )
            if decomposition_mode == DecompositionMode.GHERKIN
            else GrammaticalCompositionTools(
                analysis=analysis,
                method_searcher=method_searcher,
                class_searcher=class_searcher,
                structured_llm=structured_llm,
                project_root=str(resolved_project_root),
                module_root=resolved_module_root,
                nl2_input=nl2_input,
            )
        )

        tools, allow_duplicate_tools = tool_builder.all()

        self.nl2_input = nl2_input
        self.decomposition_mode = decomposition_mode
        self.test_base_dir = test_base_dir

        chat_prompt, system_prompt = self._init_prompts()
        self.chat_prompt = chat_prompt

        # Determine if the model supports parallel tool calls from configuration
        parallelizable: bool = bool(Config().get("llm", "can_parallel_tool"))
        max_iters = Config().get("composition", "max_iters")
        duplicate_tools_str = (
            ", ".join(f"`{t.name}`" for t in allow_duplicate_tools)
            if allow_duplicate_tools
            else ""
        )

        if self.decomposition_mode != DecompositionMode.GHERKIN:
            raise NotImplementedError("Only supported for Gherkin style...")

        system_kwargs = {
            "parallelizable": parallelizable,
            "max_iters": max_iters,
            "duplicate_tools": duplicate_tools_str,
        }

        system_message = system_prompt.format(**system_kwargs)

        self.agent = CompositionReActAgent(
            llm=decision_llm,
            tools=tools,
            allow_duplicate_tools=allow_duplicate_tools,
            system_message=system_message,
            project_root=resolved_project_root,
            test_base_dir=test_base_dir,
            module_root=resolved_module_root,
            max_iters=max_iters,
            parallelizable=parallelizable,
        )

    def _init_prompts(self) -> tuple[PromptTemplate, PromptTemplate]:
        # Choose prompts according to decomposition mode
        if self.decomposition_mode == DecompositionMode.GHERKIN:
            chat_file = "composition_agent_gherkin.jinja2"
            system_file = "composition_agent_gherkin.jinja2"
        else:
            chat_file = "composition_agent_grammatical.jinja2"
            system_file = "composition_agent_grammatical.jinja2"

        chat_prompt = LoadPrompt.load_prompt(
            chat_file, PromptFormat.JINJA2, prompt_type="chat"
        )
        system_prompt = LoadPrompt.load_prompt(
            system_file, PromptFormat.JINJA2, prompt_type="system"
        )
        return chat_prompt, system_prompt

    def reset_agent(self) -> None:
        self.agent.reset_agent()

    # Shared signature implemented by subclasses. Intentionally untyped for different decomposition modes
    def assign_task(
        self, blocks, *, instructions: str, agent_state: AgentState | None = None
    ):
        raise NotImplementedError
