from __future__ import annotations

from pathlib import Path
from typing import Any

from cldk.analysis.java import JavaAnalysis
from langchain_core.prompts import PromptTemplate

from nltest.nl2test.models import NL2TestInput
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.nl2test.preprocessing.searchers import ClassSearcher, MethodSearcher
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.utils.llm import ClientType, LLMClient, UsageTracker
from nltest.utils.config import Config
from nltest.nl2test.generation.supervisor.agent import SupervisorReActAgent
from nltest.nl2test.generation.supervisor.tools import (
    GherkinSupervisorTools,
    GrammaticalSupervisorTools,
)
from nltest.nl2test.generation.localization.orchestrators.gherkin import (
    GherkinLocalizationOrchestrator,
)
from nltest.nl2test.generation.localization.orchestrators.grammatical import (
    GrammaticalLocalizationOrchestrator,
)
from nltest.nl2test.generation.composition.orchestrators.gherkin import (
    GherkinCompositionOrchestrator,
)
from nltest.nl2test.generation.composition.orchestrators.grammatical import (
    GrammaticalCompositionOrchestrator,
)


class BaseSupervisorOrchestrator:
    def __init__(
        self,
        *,
        analysis: JavaAnalysis,
        method_searcher: MethodSearcher,
        class_searcher: ClassSearcher,
        nl2_input: NL2TestInput,
        decomposition_mode: DecompositionMode,
        base_project_dir: str,
        test_base_dir: str | Path | None = None,
        module_root: str | Path | None = None,
        usage_tracker: UsageTracker | None = None,
    ) -> None:
        # Only GHERKIN is supported for Supervisor orchestration right now.
        if decomposition_mode != DecompositionMode.GHERKIN:
            raise NotImplementedError(
                "Supervisor orchestrator currently supports only GHERKIN mode."
            )

        self.analysis = analysis
        self.method_searcher = method_searcher
        self.class_searcher = class_searcher
        self.nl2_input = nl2_input
        self.decomposition_mode = decomposition_mode
        self.base_project_dir = base_project_dir
        self.test_base_dir = test_base_dir
        self.module_root = module_root

        self.usage_tracker = usage_tracker or UsageTracker()

        # Validate required dependencies are present
        if self.method_searcher is None or self.class_searcher is None:
            raise Exception(
                "The database is not indexed and provided to the supervisor orchestrator."
            )

        # Initialize LLMs and tools
        decision_llm = LLMClient(
            ClientType.DECISION,
            usage_tracker=self.usage_tracker,
        )

        resolved_project_root = Path(base_project_dir).expanduser().resolve()
        resolved_module_root = None
        if module_root is not None:
            resolved_module_root = Path(module_root).expanduser()
            if not resolved_module_root.is_absolute():
                resolved_module_root = resolved_project_root / resolved_module_root
            resolved_module_root = resolved_module_root.resolve()

        # Choose tool builder by decomposition mode (guarded above, but keep structure)
        if self.decomposition_mode == DecompositionMode.GHERKIN:
            tool_builder = GherkinSupervisorTools(
                llm=decision_llm,
                project_root=str(resolved_project_root),
            )
            localization_agent = GherkinLocalizationOrchestrator(
                analysis=analysis,
                method_searcher=method_searcher,
                class_searcher=class_searcher,
                nl2_input=nl2_input,
                usage_tracker=self.usage_tracker,
            )
            composition_agent = GherkinCompositionOrchestrator(
                analysis=analysis,
                method_searcher=method_searcher,
                class_searcher=class_searcher,
                nl2_input=nl2_input,
                project_root=str(resolved_project_root),
                test_base_dir=test_base_dir,
                module_root=resolved_module_root,
                usage_tracker=self.usage_tracker,
            )
        else:
            tool_builder = GrammaticalSupervisorTools(
                llm=decision_llm,
                project_root=str(resolved_project_root),
            )
            localization_agent = GrammaticalLocalizationOrchestrator(
                analysis=analysis,
                method_searcher=method_searcher,
                class_searcher=class_searcher,
                nl2_input=nl2_input,
                usage_tracker=self.usage_tracker,
            )
            composition_agent = GrammaticalCompositionOrchestrator(
                analysis=analysis,
                method_searcher=method_searcher,
                class_searcher=class_searcher,
                nl2_input=nl2_input,
                project_root=str(resolved_project_root),
                test_base_dir=test_base_dir,
                module_root=resolved_module_root,
                usage_tracker=self.usage_tracker,
            )

        tools, allow_duplicate_tools = tool_builder.all()

        # Prompts
        chat_prompt, system_prompt = self._init_prompts()
        self.chat_prompt = chat_prompt

        # System prompt variables
        parallelizable: bool = bool(Config().get("llm", "can_parallel_tool"))
        max_iters = Config().get("supervisor", "max_iters")
        duplicate_tools_str = (
            ", ".join(f"`{t.name}`" for t in allow_duplicate_tools)
            if allow_duplicate_tools
            else ""
        )

        system_message = system_prompt.format(
            max_iters=max_iters, duplicate_tools=duplicate_tools_str
        )

        # Initialize the Supervisor ReAct agent
        self.agent = SupervisorReActAgent(
            llm=decision_llm,
            tools=tools,
            allow_duplicate_tools=allow_duplicate_tools,
            system_message=system_message,
            nl_description=self.nl2_input.description if self.nl2_input else "",
            project_root=str(resolved_project_root),
            test_base_dir=test_base_dir,
            module_root=resolved_module_root,
            max_iters=max_iters,
            localization_agent=localization_agent,
            composition_agent=composition_agent,
            parallelizable=parallelizable,
        )

    def assign_task(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def _init_prompts(self) -> tuple[PromptTemplate, PromptTemplate]:
        # Only GHERKIN supported for supervision currently
        chat_file = "supervisor_agent_gherkin.jinja2"
        system_file = "supervisor_agent_gherkin.jinja2"

        chat_prompt = LoadPrompt.load_prompt(
            chat_file, PromptFormat.JINJA2, prompt_type="chat"
        )
        system_prompt = LoadPrompt.load_prompt(
            system_file, PromptFormat.JINJA2, prompt_type="system"
        )
        return chat_prompt, system_prompt
