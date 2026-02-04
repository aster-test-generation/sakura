from __future__ import annotations
_A=None
from pathlib import Path
from typing import Any
from cldk.analysis.java import JavaAnalysis
from langchain_core.prompts import PromptTemplate
from sakura.nl2test.models import NL2TestInput
from sakura.nl2test.models.decomposition import DecompositionMode
from sakura.nl2test.preprocessing.searchers import ClassSearcher,MethodSearcher
from sakura.nl2test.prompts.load_prompt import LoadPrompt,PromptFormat
from sakura.utils.llm import ClientType,LLMClient,UsageTracker
from sakura.utils.config import Config
from sakura.nl2test.generation.supervisor.agent import SupervisorReActAgent
from sakura.nl2test.generation.supervisor.tools import GherkinSupervisorTools,GrammaticalSupervisorTools
from sakura.nl2test.generation.localization.orchestrators.gherkin import GherkinLocalizationOrchestrator
from sakura.nl2test.generation.localization.orchestrators.grammatical import GrammaticalLocalizationOrchestrator
from sakura.nl2test.generation.composition.orchestrators.gherkin import GherkinCompositionOrchestrator
from sakura.nl2test.generation.composition.orchestrators.grammatical import GrammaticalCompositionOrchestrator
class BaseSupervisorOrchestrator:
	def __init__(A,*,analysis:JavaAnalysis,method_searcher:MethodSearcher,class_searcher:ClassSearcher,nl2_input:NL2TestInput,decomposition_mode:DecompositionMode,base_project_dir:str,test_base_dir:str|Path|_A=_A,module_root:str|Path|_A=_A,usage_tracker:UsageTracker|_A=_A)->_A:
		M=base_project_dir;L=decomposition_mode;I=module_root;H=test_base_dir;G=nl2_input;F=class_searcher;E=method_searcher;D=analysis
		if L!=DecompositionMode.GHERKIN:raise NotImplementedError('Supervisor orchestrator currently supports only GHERKIN mode.')
		A.analysis=D;A.method_searcher=E;A.class_searcher=F;A.nl2_input=G;A.decomposition_mode=L;A.base_project_dir=M;A.test_base_dir=H;A.module_root=I;A.usage_tracker=usage_tracker or UsageTracker()
		if A.method_searcher is _A or A.class_searcher is _A:raise Exception('The database is not indexed and provided to the supervisor orchestrator.')
		J=LLMClient(ClientType.DECISION,usage_tracker=A.usage_tracker);C=Path(M).expanduser().resolve();B=_A
		if I is not _A:
			B=Path(I).expanduser()
			if not B.is_absolute():B=C/B
			B=B.resolve()
		if A.decomposition_mode==DecompositionMode.GHERKIN:N=GherkinSupervisorTools(llm=J,project_root=str(C));O=GherkinLocalizationOrchestrator(analysis=D,method_searcher=E,class_searcher=F,nl2_input=G,usage_tracker=A.usage_tracker);P=GherkinCompositionOrchestrator(analysis=D,method_searcher=E,class_searcher=F,nl2_input=G,project_root=str(C),test_base_dir=H,module_root=B,usage_tracker=A.usage_tracker)
		else:N=GrammaticalSupervisorTools(llm=J,project_root=str(C));O=GrammaticalLocalizationOrchestrator(analysis=D,method_searcher=E,class_searcher=F,nl2_input=G,usage_tracker=A.usage_tracker);P=GrammaticalCompositionOrchestrator(analysis=D,method_searcher=E,class_searcher=F,nl2_input=G,project_root=str(C),test_base_dir=H,module_root=B,usage_tracker=A.usage_tracker)
		R,K=N.all();S,T=A._init_prompts();A.chat_prompt=S;U:bool=bool(Config().get('llm','can_parallel_tool'));Q=Config().get('supervisor','max_iters');V=', '.join(f"`{A.name}`"for A in K)if K else'';W=T.format(max_iters=Q,duplicate_tools=V);A.agent=SupervisorReActAgent(llm=J,tools=R,allow_duplicate_tools=K,system_message=W,nl_description=A.nl2_input.description if A.nl2_input else'',project_root=str(C),test_base_dir=H,module_root=B,max_iters=Q,localization_agent=O,composition_agent=P,parallelizable=U)
	def assign_task(A,*B:Any,**C:Any):raise NotImplementedError
	def _init_prompts(F)->tuple[PromptTemplate,PromptTemplate]:A='supervisor_agent_gherkin.jinja2';B=A;C=A;D=LoadPrompt.load_prompt(B,PromptFormat.JINJA2,prompt_type='chat');E=LoadPrompt.load_prompt(C,PromptFormat.JINJA2,prompt_type='system');return D,E