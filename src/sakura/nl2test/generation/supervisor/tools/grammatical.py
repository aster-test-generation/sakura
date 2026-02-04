from __future__ import annotations
_A='instructions'
from pathlib import Path
import textwrap
from typing import Union
from langchain_core.tools import BaseTool
from.base import BaseSupervisorTools
from sakura.nl2test.core.deferred_tool import DeferredTool
from sakura.nl2test.models import CallAgentGrammaticalArgs
from sakura.nl2test.generation.supervisor.tool_descriptions import CALL_LOCALIZATION_AGENT_GRAMMATICAL_DESC,CALL_COMPOSITION_AGENT_GRAMMATICAL_DESC
from sakura.utils.llm import LLMClient
class GrammaticalSupervisorTools(BaseSupervisorTools):
	'\n    Tool builder for Grammatical-mode supervisor agent.\n\n    Extends base tools with Grammatical-specific delegation tools:\n    - call_localization_agent: Delegate to localization agent (deferred)\n    - call_composition_agent: Delegate to composition agent (deferred)\n    '
	def __init__(A,*,llm:LLMClient,project_root:Union[str,Path])->None:super().__init__(llm=llm,project_root=project_root);B=A._make_call_localization_agent_tool();C=A._make_call_composition_agent_tool();A.tools.append(B);A.tools.append(C);A.allow_duplicate_tools.append(B);A.allow_duplicate_tools.append(C)
	def _make_call_localization_agent_tool(A)->BaseTool:"\n        Create the call_localization_agent tool for Grammatical mode.\n\n        This is a deferred tool - it returns the instructions and the agent's\n        process_tool_output hook invokes the actual localization orchestrator.\n        ";return DeferredTool.create(name='call_localization_agent',description=textwrap.dedent(CALL_LOCALIZATION_AGENT_GRAMMATICAL_DESC).strip(),args_schema=CallAgentGrammaticalArgs,returns_input_keys=[_A],processing_note='Agent invokes localization orchestrator with instructions, injecting current AtomicBlockList from state')
	def _make_call_composition_agent_tool(A)->BaseTool:"\n        Create the call_composition_agent tool for Grammatical mode.\n\n        This is a deferred tool - it returns the instructions and the agent's\n        process_tool_output hook invokes the actual composition orchestrator.\n        ";return DeferredTool.create(name='call_composition_agent',description=textwrap.dedent(CALL_COMPOSITION_AGENT_GRAMMATICAL_DESC).strip(),args_schema=CallAgentGrammaticalArgs,returns_input_keys=[_A],processing_note='Agent invokes composition orchestrator with instructions, injecting current AtomicBlockList from state')