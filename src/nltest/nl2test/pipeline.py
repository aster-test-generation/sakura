from typing import Tuple

from cldk.analysis.java import JavaAnalysis

from nltest.nl2test.agents.supervisor_agent import SupervisorAgent
from nltest.nl2test.model.models import AgentState
from nltest.nl2test.preprocessing.indexers.class_indexer import ClassIndexer
from nltest.nl2test.preprocessing.indexers.method_indexer import MethodIndexer
from nltest.nl2test.preprocessing.searchers.class_searcher import ClassSearcher
from nltest.nl2test.preprocessing.searchers.method_searcher import MethodSearcher


class Pipeline:
    def __init__(self, analysis: JavaAnalysis):
        self.method_indexer = MethodIndexer(analysis)
        self.analysis = analysis

    def run_preprocess(self) -> Tuple[MethodSearcher, ClassSearcher]:
        method_searcher = MethodIndexer(self.analysis).build_index()
        class_searcher = ClassIndexer(self.analysis).build_index()
        return method_searcher, class_searcher

    def run_agents(self, nl_description: str) -> str:
        method_searcher, class_searcher = self.run_preprocess()
        supervisor = SupervisorAgent(self.analysis, searcher)
        state = AgentState(messages=[], iterations=0)
        result = supervisor.agent.invoke(nl_description, state=state)
        final_test = result['messages'][-1].content
        return final_test