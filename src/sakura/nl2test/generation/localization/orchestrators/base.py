from __future__ import annotations
_A=None
from typing import List,Tuple
from cldk.analysis.java import JavaAnalysis
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from sakura.nl2test.generation.localization.agent import LocalizationReActAgent
from sakura.nl2test.generation.localization.tools import GherkinLocalizationTools,GrammaticalLocalizationTools
from sakura.nl2test.models import AgentState,NL2TestInput
from sakura.nl2test.models.decomposition import DecompositionMode
from sakura.nl2test.prompts.load_prompt import LoadPrompt,PromptFormat
from sakura.nl2test.preprocessing.searchers import ClassSearcher,MethodSearcher
from sakura.utils.config import Config
from sakura.utils.llm import ClientType,LLMClient,UsageTracker
class BaseLocalizationOrchestrator:
	def __init__(A,*,analysis:JavaAnalysis,method_searcher:MethodSearcher,class_searcher:ClassSearcher,nl2_input:NL2TestInput,decomposition_mode:DecompositionMode,usage_tracker:UsageTracker|_A=_A)->_A:
		E=class_searcher;D=method_searcher;C=analysis;B=decomposition_mode;A.usage_tracker=usage_tracker or UsageTracker();I=LLMClient(ClientType.DECISION,usage_tracker=A.usage_tracker)
		if B==DecompositionMode.GHERKIN:F=GherkinLocalizationTools(analysis=C,method_searcher=D,class_searcher=E)
		else:F=GrammaticalLocalizationTools(analysis=C,method_searcher=D,class_searcher=E)
		J,K=F.all();A.nl2_input=nl2_input;A.decomposition_mode=B;L,M=A._init_prompts();A.chat_prompt=L;G:bool=bool(Config().get('llm','can_parallel_tool'));H=Config().get('localization','max_iters');N=M.format(parallelizable=G,max_iters=H);A.agent=LocalizationReActAgent(llm=I,tools=J,allow_duplicate_tools=K,max_iters=H,system_message=N,decomposition_mode=B,parallelizable=G)
	def _init_prompts(E)->tuple[PromptTemplate,PromptTemplate]:
		D='localization_agent_grammatical.jinja2';C='localization_agent_gherkin.jinja2'
		if E.decomposition_mode==DecompositionMode.GHERKIN:A=C;B=C
		else:A=D;B=D
		F=LoadPrompt.load_prompt(A,PromptFormat.JINJA2,prompt_type='chat');G=LoadPrompt.load_prompt(B,PromptFormat.JINJA2,prompt_type='system');return F,G
	def reset_agent(A)->_A:A.agent.reset_agent()
	def assign_task(A,blocks,*,instructions:str,agent_state:AgentState|_A=_A):raise NotImplementedError