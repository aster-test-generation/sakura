from typing import List, Tuple

from cldk.analysis.java import JavaAnalysis
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver

from nltest.nl2test.generation.localization.agent import LocalizationReActAgent
from nltest.nl2test.generation.localization.tools import LocalizationTools
from nltest.nl2test.model.models import AgentState, AtomicBlock, NL2TestInput
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.nl2test.preprocessing.searchers import ClassSearcher
from nltest.nl2test.preprocessing.searchers import MethodSearcher
from nltest.utils.llm.llm_client import LLMClient, ClientType
from nltest.utils.config.config import Config


class LocalizationOrchestrator:
    def __init__(self, analysis: JavaAnalysis, method_searcher: MethodSearcher,
                 class_searcher: ClassSearcher, nl2_input: NL2TestInput):
        decision_llm = LLMClient(ClientType.DECISION)
        structured_llm = LLMClient(ClientType.STRUCTURED)

        tools: List[BaseTool] = LocalizationTools(
            analysis=analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            structured_llm=structured_llm,
        ).all()

        self.nl2_input = nl2_input

        self.chat_prompt = LoadPrompt().load_prompt("localization_agent.jinja2", PromptFormat.JINJA2, prompt_type="chat")

        self.agent = LocalizationReActAgent(
            llm=decision_llm,
            tools=tools,
            max_iters=Config().get("localization", "max_iters")
        )

    def assign_task(self, instructions: str, atomic_blocks: List[AtomicBlock]) -> Tuple[List[AtomicBlock], str]:
        """
        Instruct the agent on how or why the atomic_blocks should be updated, and return the result.

        Args:
            instructions: What changes to make, or criticisms with the current decomposition.
            atomic_blocks: The current list of AtomicBlock objects to be modified and improved.

        Returns:
            The revised list of AtomicBlock objects.
        """
        initial_state = AgentState(
            atomic_blocks=atomic_blocks,
        )

        chat_prompt = self.chat_prompt.format(
            nl_description=self.nl2_input.description,
            instructions=instructions,
            atomic_blocks=atomic_blocks,
        )

        updated_state: AgentState = self.agent.invoke(chat_prompt, initial_state)
        return updated_state.atomic_blocks, (updated_state.final_comments or "No comments.")
