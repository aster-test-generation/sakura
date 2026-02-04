from __future__ import annotations
_A=None
from pathlib import Path
from typing import Tuple
from cldk.analysis.java import JavaAnalysis
from sakura.nl2test.preprocessing.searchers import MethodSearcher,ClassSearcher
from sakura.nl2test.models.decomposition import DecompositionMode
from sakura.nl2test.models import AgentState,LocalizedScenario,NL2TestInput
from sakura.utils.llm import UsageTracker
from.base import BaseSupervisorOrchestrator
class GherkinSupervisorOrchestrator(BaseSupervisorOrchestrator):
	def __init__(A,*,analysis:JavaAnalysis,method_searcher:MethodSearcher,class_searcher:ClassSearcher,nl2_input:NL2TestInput,base_project_dir:str,test_base_dir:str|Path|_A=_A,module_root:str|Path|_A=_A,usage_tracker:UsageTracker|_A=_A)->_A:super().__init__(analysis=analysis,method_searcher=method_searcher,class_searcher=class_searcher,nl2_input=nl2_input,decomposition_mode=DecompositionMode.GHERKIN,base_project_dir=base_project_dir,test_base_dir=test_base_dir,module_root=module_root,usage_tracker=usage_tracker)
	def assign_task(A,blocks)->Tuple[AgentState,AgentState,AgentState]:
		if A.decomposition_mode!=DecompositionMode.GHERKIN:raise TypeError('GherkinSupervisorOrchestrator is not in GHERKIN mode.')
		C:LocalizedScenario=blocks;D=AgentState(localized_scenario=C);E=A.chat_prompt.format(blocks=C,nl_description=A.nl2_input.description);B:AgentState=A.agent.invoke(E,D,config={'configurable':{'thread_id':f"sup:{A.nl2_input.id}"}})
		if B.curr_tool_trajectory:B.tool_trajectories.append(B.curr_tool_trajectory.copy());B.curr_tool_trajectory.clear()
		F:AgentState=A.agent.localization_state;G:AgentState=A.agent.composition_state;return B,F,G