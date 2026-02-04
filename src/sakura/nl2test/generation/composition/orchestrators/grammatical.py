from __future__ import annotations
_A=None
from pathlib import Path
from sakura.nl2test.generation.composition.orchestrators.base import BaseCompositionOrchestrator
from cldk.analysis.java import JavaAnalysis
from sakura.nl2test.preprocessing.searchers import MethodSearcher,ClassSearcher
from sakura.nl2test.models import AgentState,AtomicBlockList,NL2TestInput
from sakura.nl2test.models.decomposition import DecompositionMode
from sakura.utils.llm import UsageTracker
class GrammaticalCompositionOrchestrator(BaseCompositionOrchestrator):
	def __init__(A,*,analysis:JavaAnalysis,method_searcher:MethodSearcher,class_searcher:ClassSearcher,nl2_input:NL2TestInput,project_root:str,test_base_dir:str|Path|_A=_A,module_root:str|Path|_A=_A,usage_tracker:UsageTracker|_A=_A)->_A:super().__init__(analysis=analysis,method_searcher=method_searcher,class_searcher=class_searcher,nl2_input=nl2_input,project_root=project_root,test_base_dir=test_base_dir,module_root=module_root,decomposition_mode=DecompositionMode.GRAMMATICAL,usage_tracker=usage_tracker)
	def assign_task(A,blocks,*,instructions:str,agent_state:AgentState|_A=_A)->AgentState:
		B=agent_state
		if A.decomposition_mode!=DecompositionMode.GRAMMATICAL:raise TypeError('GrammaticalCompositionOrchestrator is not in GRAMMATICAL mode.')
		C:AtomicBlockList=blocks
		if B is not _A:D=B.model_copy(deep=True)if hasattr(B,'model_copy')else B;D.atomic_blocks=C
		else:D=AgentState(atomic_blocks=C)
		E=A.chat_prompt.format(nl_description=A.nl2_input.description,instructions=instructions,atomic_blocks=C.atomic_blocks);F:AgentState=A.agent.invoke(E,D,config={'configurable':{'thread_id':f"cmp:{A.nl2_input.id}"}});return F