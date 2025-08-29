from typing import List, Tuple

from cldk.analysis.java import JavaAnalysis
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver

from nltest.nl2test.generation.composition.agent import CompositionReActAgent
from nltest.nl2test.generation.composition.tools import CompositionTools
from nltest.nl2test.models import AgentState, AtomicBlock, AtomicBlockList, NL2TestInput
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.nl2test.preprocessing.searchers import ClassSearcher
from nltest.nl2test.preprocessing.searchers import MethodSearcher
from nltest.utils.llm.llm_client import LLMClient, ClientType
from nltest.utils.config.config import Config


class CompositionOrchestrator:
    def __init__(self, analysis: JavaAnalysis, method_searcher: MethodSearcher,
                 class_searcher: ClassSearcher, nl2_input: NL2TestInput):
        decision_llm = LLMClient(ClientType.DECISION)
        structured_llm = LLMClient(ClientType.STRUCTURED)

        tools: List[BaseTool] = CompositionTools(
            analysis=analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            structured_llm=structured_llm,
            base_project_dir=".",  # This should be the actual project root
            nl2_input=nl2_input,
        ).all()

        self.nl2_input = nl2_input

        self.chat_prompt = LoadPrompt().load_prompt("composition_agent.jinja2", PromptFormat.JINJA2, prompt_type="chat")

        self.agent = CompositionReActAgent(
            llm=decision_llm,
            tools=tools,
            max_iters=Config().get("composition", "max_iters")
        )

    def assign_task(self, instructions: str, atomic_blocks: AtomicBlockList) -> Tuple[AtomicBlockList, str]:
        """
        Instruct the agent on how or why the atomic_blocks should be updated, and return the result.

        Args:
            instructions: What changes to make, or criticisms with the current decomposition.
            atomic_blocks: The current AtomicBlockList to be modified and improved.

        Returns:
            A tuple containing the revised AtomicBlockList and any final comments.
        """
        initial_state = AgentState(
            atomic_blocks=atomic_blocks,
        )

        chat_prompt = self.chat_prompt.format(
            nl_description=self.nl2_input.description,
            instructions=instructions,
            atomic_blocks=atomic_blocks.atomic_blocks,
        )

        updated_state: AgentState = self.agent.invoke(chat_prompt, initial_state)
        return updated_state.atomic_blocks, (updated_state.final_comments or "No comments.")
