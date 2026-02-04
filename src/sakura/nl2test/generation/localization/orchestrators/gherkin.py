from __future__ import annotations
_A=None
from sakura.nl2test.generation.localization.orchestrators.base import BaseLocalizationOrchestrator
from cldk.analysis.java import JavaAnalysis
from sakura.nl2test.preprocessing.searchers import MethodSearcher,ClassSearcher
from sakura.nl2test.models import AgentState,LocalizedScenario,NL2TestInput
from sakura.utils.llm import UsageTracker
from sakura.nl2test.models.decomposition import DecompositionMode
class GherkinLocalizationOrchestrator(BaseLocalizationOrchestrator):
	def __init__(A,*,analysis:JavaAnalysis,method_searcher:MethodSearcher,class_searcher:ClassSearcher,nl2_input:NL2TestInput,usage_tracker:UsageTracker|_A=_A)->_A:super().__init__(analysis=analysis,method_searcher=method_searcher,class_searcher=class_searcher,nl2_input=nl2_input,decomposition_mode=DecompositionMode.GHERKIN,usage_tracker=usage_tracker)
	def assign_task(A,blocks,*,instructions:str,agent_state:AgentState|_A=_A)->AgentState:
		B=agent_state
		if A.decomposition_mode!=DecompositionMode.GHERKIN:raise TypeError('GherkinLocalizationOrchestrator is not in GHERKIN mode.')
		C:LocalizedScenario=blocks
		if B is not _A:D=B.model_copy(deep=True)if hasattr(B,'model_copy')else B;D.localized_scenario=C
		else:D=AgentState(localized_scenario=C)
		E=A.chat_prompt.format(nl_description=A.nl2_input.description,instructions=instructions,steps=C);F:AgentState=A.agent.invoke(E,D,config={'configurable':{'thread_id':f"loc:{A.nl2_input.id}"}});return F