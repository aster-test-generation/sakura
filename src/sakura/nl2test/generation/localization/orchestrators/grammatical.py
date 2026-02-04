from __future__ import annotations
_A=None
from sakura.nl2test.generation.localization.orchestrators.base import BaseLocalizationOrchestrator
from cldk.analysis.java import JavaAnalysis
from sakura.nl2test.preprocessing.searchers import MethodSearcher,ClassSearcher
from sakura.nl2test.models import AgentState,AtomicBlockList,NL2TestInput
from sakura.nl2test.models.decomposition import DecompositionMode
from sakura.utils.llm import UsageTracker
class GrammaticalLocalizationOrchestrator(BaseLocalizationOrchestrator):
	def __init__(A,*,analysis:JavaAnalysis,method_searcher:MethodSearcher,class_searcher:ClassSearcher,nl2_input:NL2TestInput,usage_tracker:UsageTracker|_A=_A)->_A:super().__init__(analysis=analysis,method_searcher=method_searcher,class_searcher=class_searcher,nl2_input=nl2_input,decomposition_mode=DecompositionMode.GRAMMATICAL,usage_tracker=usage_tracker)
	def assign_task(A,blocks,*,instructions:str,agent_state:AgentState|_A=_A)->AgentState:
		B=agent_state
		if A.decomposition_mode!=DecompositionMode.GRAMMATICAL:raise TypeError('GrammaticalLocalizationOrchestrator is not in GRAMMATICAL mode.')
		C:AtomicBlockList=blocks
		if B is not _A:D=B.model_copy(deep=True)if hasattr(B,'model_copy')else B;D.atomic_blocks=C
		else:D=AgentState(atomic_blocks=C)
		E=C.atomic_blocks;F=A.chat_prompt.format(nl_description=A.nl2_input.description,instructions=instructions,blocks=E);G:AgentState=A.agent.invoke(F,D,config={'configurable':{'thread_id':f"loc:{A.nl2_input.id}"}});return G