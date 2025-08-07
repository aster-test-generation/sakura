from typing import List

from cldk.analysis.java import JavaAnalysis
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver

from nltest.nl2test.generation.composition.agent import CompositionReActAgent
from nltest.nl2test.generation.composition.tools import CompositionTools
from nltest.nl2test.model.models import AgentState, AtomicBlock
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.nl2test.preprocessing.searchers.class_searcher import ClassSearcher
from nltest.nl2test.preprocessing.searchers.method_searcher import MethodSearcher
from nltest.utils.llm.llm_client import LLMClient, ClientType


class CompositionOrchestrator:
    def __init__(self, analysis: JavaAnalysis, method_searcher: MethodSearcher,
                 class_searcher: ClassSearcher, nl_description: str):
        # TODO: Change the nl_description input into the Test2NL object
        llm = LLMClient(ClientType.DECISION)

        tools: List[BaseTool] = CompositionTools(
            analysis=analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            llm=llm,
        ).all()

        agent_prompt = LoadPrompt.load_prompt("composition_agent.jinja2", PromptFormat.JINJA2)

        self.agent = CompositionReActAgent(
            llm=llm,
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
