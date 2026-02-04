from __future__ import annotations
from pathlib import Path
import textwrap
from typing import Union
from langchain_core.tools import BaseTool
from.base import BaseCompositionTools
from sakura.nl2test.core.deferred_tool import DeferredTool
from sakura.nl2test.models import ModifyAtomicBlockNoteArgs,NL2TestInput
from cldk.analysis.java import JavaAnalysis
from sakura.nl2test.preprocessing.searchers import MethodSearcher,ClassSearcher
from sakura.utils.llm import LLMClient
from sakura.nl2test.generation.composition.tool_descriptions import MODIFY_ATOMIC_BLOCK_NOTE_DESC
class GrammaticalCompositionTools(BaseCompositionTools):
	'\n    Tool builder for Grammatical-mode composition agent.\n\n    Extends base composition tools with Grammatical-specific tools:\n    - modify_scenario_comment: Update note on an atomic block (deferred)\n    '
	def __init__(A,*,analysis:JavaAnalysis,method_searcher:MethodSearcher,class_searcher:ClassSearcher,structured_llm:LLMClient,project_root:Union[str,Path],module_root:Union[str,Path]|None=None,nl2_input:NL2TestInput):super().__init__(analysis=analysis,method_searcher=method_searcher,class_searcher=class_searcher,structured_llm=structured_llm,project_root=str(project_root),module_root=module_root,nl2_input=nl2_input);A.tools.append(A._make_modify_scenario_comment_tool())
	def _make_modify_scenario_comment_tool(A)->BaseTool:"\n        Create the modify_scenario_comment tool for Grammatical mode.\n\n        This is a deferred tool - it returns the block order and note,\n        and the agent's process_tool_output hook updates the atomic blocks in state.\n        ";return DeferredTool.create(name='modify_scenario_comment',description=textwrap.dedent(MODIFY_ATOMIC_BLOCK_NOTE_DESC).strip(),args_schema=ModifyAtomicBlockNoteArgs,returns_input_keys=['order','note'],processing_note='Agent locates block by order in state.atomic_blocks and updates its notes field')