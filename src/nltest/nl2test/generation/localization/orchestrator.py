from typing import List, Tuple, Union, overload

from cldk.analysis.java import JavaAnalysis
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver

from nltest.nl2test.generation.localization.agent import LocalizationReActAgent
from nltest.nl2test.generation.localization.tools import LocalizationTools
from nltest.nl2test.models import (
    AgentState,
    AtomicBlockList,
    GrammaticalBlockList,
    NL2TestInput,
    Scenario,
    LocalizedScenario,
)
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from langchain_core.prompts import PromptTemplate
from nltest.nl2test.preprocessing.searchers import ClassSearcher
from nltest.nl2test.preprocessing.searchers import MethodSearcher
from nltest.utils.llm.llm_client import LLMClient, ClientType
from nltest.utils.config.config import Config


class LocalizationOrchestrator:
    def __init__(
        self,
        analysis: JavaAnalysis,
        method_searcher: MethodSearcher,
        class_searcher: ClassSearcher,
        nl2_input: NL2TestInput,
        *,
        decomposition_mode: DecompositionMode = DecompositionMode.GRAMMATICAL,
    ):
        decision_llm = LLMClient(ClientType.DECISION)
        structured_llm = LLMClient(ClientType.STRUCTURED)

        tools: List[BaseTool] = LocalizationTools(
            analysis=analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            structured_llm=structured_llm,
            decomposition_mode=decomposition_mode,
        ).all()

        self.nl2_input = nl2_input
        self.decomposition_mode = decomposition_mode

        # Initialize prompts (chat + system) based on decomposition mode
        chat_prompt, system_prompt = self._init_prompts()
        self.chat_prompt = chat_prompt
        system_message = system_prompt.format()

        self.agent = LocalizationReActAgent(
            llm=decision_llm,
            tools=tools,
            max_iters=Config().get("localization", "max_iters"),
            system_message=system_message,
            decomposition_mode=self.decomposition_mode,
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

    @overload
    def assign_task(
        self, instructions: str, *, grammatical_blocks: GrammaticalBlockList
    ) -> Tuple[AtomicBlockList, str]: ...

    @overload
    def assign_task(
        self, instructions: str, *, scenario: Scenario
    ) -> Tuple[LocalizedScenario, str]: ...

    def assign_task(
        self,
        instructions: str,
        *,
        grammatical_blocks: GrammaticalBlockList | None = None,
        scenario: Scenario | None = None,
    ) -> Tuple[LocalizedScenario, str] | Tuple[AtomicBlockList, str]:
        """
        Assign a localization task with mode-specific inputs and outputs.

        - GRAMMATICAL mode: provide grammatical_blocks, returns (AtomicBlockList, comments)
        - GHERKIN mode: provide scenario, returns (LocalizedScenario, comments)
        """
        if (grammatical_blocks is None) == (scenario is None):
            raise ValueError("Provide exactly one of grammatical_blocks or scenario.")

        if self.decomposition_mode == DecompositionMode.GHERKIN:
            if scenario is None:
                raise TypeError("In GHERKIN mode, pass scenario=")
            initial_state = AgentState(scenario=scenario)
            prompt_blocks = scenario
        elif self.decomposition_mode == DecompositionMode.GRAMMATICAL:
            if grammatical_blocks is None:
                raise TypeError("In GRAMMATICAL mode, pass grammatical_blocks=")
            initial_state = AgentState(grammatical_blocks=grammatical_blocks)
            prompt_blocks = grammatical_blocks.grammatical_blocks
        else:
            raise ValueError(
                f"Unsupported decomposition mode: {self.decomposition_mode}"
            )

        chat_prompt = self.chat_prompt.format(
            nl_description=self.nl2_input.description,
            instructions=instructions,
            blocks=prompt_blocks,
        )

        updated_state: AgentState = self.agent.invoke(chat_prompt, initial_state)

        if self.decomposition_mode == DecompositionMode.GHERKIN:
            if not updated_state.localized_scenario:
                raise ValueError(
                    "Agent did not return LocalizedScenario in GHERKIN mode"
                )
            return updated_state.localized_scenario, (
                updated_state.final_comments or "No comments."
            )

        if not updated_state.atomic_blocks:
            raise ValueError("Agent did not return AtomicBlockList in GRAMMATICAL mode")
        return updated_state.atomic_blocks, (
            updated_state.final_comments or "No comments."
        )
