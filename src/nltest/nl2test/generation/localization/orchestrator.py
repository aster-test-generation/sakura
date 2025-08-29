from typing import List, Tuple, Union

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

        self.chat_prompt = LoadPrompt().load_prompt(
            "localization_agent.jinja2", PromptFormat.JINJA2, prompt_type="chat"
        )

        self.agent = LocalizationReActAgent(
            llm=decision_llm,
            tools=tools,
            max_iters=Config().get("localization", "max_iters"),
        )

    def assign_task(
        self, instructions: str, blocks: Union[GrammaticalBlockList, Scenario]
    ) -> Tuple[Union[AtomicBlockList, LocalizedScenario], str]:
        """
        Instruct the agent on how or why the blocks should be updated, and return the result.

        Args:
            instructions: What changes to make, or criticisms with the current decomposition.
            blocks: Either the current GrammaticalBlockList (grammatical) or a Scenario (gherkin).

        Returns:
            - GRAMMATICAL mode: AtomicBlockList and final comments
            - GHERKIN mode: LocalizedScenario and final comments
        """
        if self.decomposition_mode == DecompositionMode.GHERKIN:
            if not isinstance(blocks, Scenario):
                raise TypeError("In GHERKIN mode, blocks must be a Scenario")
            initial_state = AgentState(scenario=blocks)
            prompt_blocks = blocks
        elif self.decomposition_mode == DecompositionMode.GRAMMATICAL:
            if not isinstance(blocks, GrammaticalBlockList):
                raise TypeError(
                    "In GRAMMATICAL mode, blocks must be a GrammaticalBlockList"
                )
            initial_state = AgentState(grammatical_blocks=blocks)
            prompt_blocks = blocks.grammatical_blocks
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

        # Gherkin mode
        if self.decomposition_mode == DecompositionMode.GHERKIN:
            if updated_state.localized_scenario is None:
                raise ValueError(
                    "Agent did not return a LocalizedScenario in GHERKIN mode"
                )
            if not isinstance(updated_state.localized_scenario, LocalizedScenario):
                raise TypeError("Expected LocalizedScenario from agent in GHERKIN mode")
            return updated_state.localized_scenario, (
                updated_state.final_comments or "No comments."
            )

        # Grammatical mode
        if updated_state.atomic_blocks is None:
            raise ValueError("Agent did not return AtomicBlockList in GRAMMATICAL mode")

        return updated_state.atomic_blocks, (
            updated_state.final_comments or "No comments."
        )
