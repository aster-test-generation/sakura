from __future__ import annotations
_A=None
from pathlib import Path
from typing import List
from cldk.analysis.java import JavaAnalysis
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from sakura.nl2test.generation.composition.agent import CompositionReActAgent
from sakura.nl2test.generation.composition.tools import GherkinCompositionTools,GrammaticalCompositionTools
from sakura.nl2test.models import AgentState,NL2TestInput
from sakura.nl2test.models.decomposition import DecompositionMode
from sakura.nl2test.prompts.load_prompt import LoadPrompt,PromptFormat
from sakura.nl2test.preprocessing.searchers import ClassSearcher,MethodSearcher
from sakura.utils.config import Config
from sakura.utils.llm import ClientType,LLMClient,UsageTracker
class BaseCompositionOrchestrator:
	def __init__(A,*,analysis:JavaAnalysis,method_searcher:MethodSearcher,class_searcher:ClassSearcher,nl2_input:NL2TestInput,project_root:str,test_base_dir:str|Path|_A=_A,module_root:str|Path|_A=_A,decomposition_mode:DecompositionMode,usage_tracker:UsageTracker|_A=_A)->_A:
		O='max_iters';K=decomposition_mode;J=module_root;I=test_base_dir;H=class_searcher;G=method_searcher;F=analysis;D=nl2_input;A.usage_tracker=usage_tracker or UsageTracker();P=LLMClient(ClientType.DECISION,usage_tracker=A.usage_tracker);L=LLMClient(ClientType.STRUCTURED,usage_tracker=A.usage_tracker);C=Path(project_root).expanduser().resolve();B=C
		if J is not _A:
			B=Path(J).expanduser()
			if not B.is_absolute():B=C/B
			B=B.resolve()
		Q=GherkinCompositionTools(analysis=F,method_searcher=G,class_searcher=H,structured_llm=L,project_root=str(C),module_root=B,nl2_input=D)if K==DecompositionMode.GHERKIN else GrammaticalCompositionTools(analysis=F,method_searcher=G,class_searcher=H,structured_llm=L,project_root=str(C),module_root=B,nl2_input=D);R,E=Q.all();A.nl2_input=D;A.decomposition_mode=K;A.test_base_dir=I;S,T=A._init_prompts();A.chat_prompt=S;M:bool=bool(Config().get('llm','can_parallel_tool'));N=Config().get('composition',O);U=', '.join(f"`{A.name}`"for A in E)if E else''
		if A.decomposition_mode!=DecompositionMode.GHERKIN:raise NotImplementedError('Only supported for Gherkin style...')
		V={'parallelizable':M,O:N,'duplicate_tools':U};W=T.format(**V);A.agent=CompositionReActAgent(llm=P,tools=R,allow_duplicate_tools=E,system_message=W,project_root=C,test_base_dir=I,module_root=B,max_iters=N,parallelizable=M)
	def _init_prompts(E)->tuple[PromptTemplate,PromptTemplate]:
		D='composition_agent_grammatical.jinja2';C='composition_agent_gherkin.jinja2'
		if E.decomposition_mode==DecompositionMode.GHERKIN:A=C;B=C
		else:A=D;B=D
		F=LoadPrompt.load_prompt(A,PromptFormat.JINJA2,prompt_type='chat');G=LoadPrompt.load_prompt(B,PromptFormat.JINJA2,prompt_type='system');return F,G
	def reset_agent(A)->_A:A.agent.reset_agent()
	def assign_task(A,blocks,*,instructions:str,agent_state:AgentState|_A=_A):raise NotImplementedError