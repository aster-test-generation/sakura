from typing import List

from cldk.analysis.java import JavaAnalysis
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver

from nltest.nl2test.generation.localization.agent import LocalizationReActAgent
from nltest.nl2test.generation.localization.tools import LocalizationTools
from nltest.nl2test.model.models import AgentState, AtomicBlock
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.nl2test.preprocessing.searchers import ClassSearcher
from nltest.nl2test.preprocessing.searchers import MethodSearcher
from nltest.utils.llm.llm_client import LLMClient, ClientType


class LocalizationOrchestrator:
    def __init__(self, analysis: JavaAnalysis, method_searcher: MethodSearcher,
                 class_searcher: ClassSearcher, nl_description: str):
        # TODO: Change the nl_description input into the Test2NL object
        decision_llm = LLMClient(ClientType.DECISION)
        structured_llm = LLMClient(ClientType.STRUCTURED)

        tools: List[BaseTool] = LocalizationTools(
            analysis=analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            structured_llm=structured_llm,
        ).all()

        agent_prompt = LoadPrompt.load_prompt("localization_agent_v6.jinja2", PromptFormat.JINJA2)

        self.agent = LocalizationReActAgent(
            llm=decision_llm,
            tools=tools,
            prompt_template=agent_prompt,
            nl_description=nl_description,
            checkpointer=MemorySaver(),
            max_iterations=10
        )

    def assign_task(self, instructions: str, atomic_blocks: List[AtomicBlock]) -> List[AtomicBlock]:
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

        updated_state: AgentState = self.agent.invoke(instructions, initial_state)
        return updated_state.atomic_blocks
