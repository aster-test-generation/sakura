from __future__ import annotations
_G='method_signature'
_F='qualified_class_name'
_E='error_type'
_D='success'
_C=False
_B=True
_A=None
import os
from pathlib import Path
from typing import Any,cast
import logging,traceback,ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from tqdm import tqdm
from nltest.nl2test import Pipeline as NL2TestPipeline
from nltest.nl2test.models import NL2TestInput
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.utils.config import init_config
from nltest.utils.formatting import ErrorFormatter
from nltest.utils.llm.model import Provider
from nltest.utils.pretty.color_logger import RichLog
from nltest.utils.vcs.git_utils import GitUtilities
@ray.remote
class NL2TestActor:
	'\n    Ray actor that initializes config, CLDK analysis, and pipeline once per project.\n    '
	def __init__(A,*,project_name:str,base_project_dir:str,base_analysis_dir:str,output_dir:str,llm_model:str,emb_model:str|_A,llm_provider:Provider|_A,llm_api_url:str|_A,emb_provider:Provider|_A,emb_api_url:str|_A,decomposition_mode:str|DecompositionMode,supervisor_max_iters:int,localization_max_iters:int,composition_max_iters:int,can_parallel_tool:bool=_B,use_stored_index:bool=_B,debug:bool=_C,log_file_name:str|_A=_A,exclude_test_dirs:bool=_C,configure_reasoning:bool=_C,reasoning_effort:str='low',exclude_reasoning:bool=_B,max_tokens:int=16384,openrouter_ignore_providers:list[str]|_A=_A,save_localized_scenarios:bool=_C)->_A:
		F=log_file_name;C=decomposition_mode;B=project_name;A.project_name=B;A.project_root=Path(base_project_dir)/B;A.analysis_dir=Path(base_analysis_dir)/B;A.project_output_dir=Path(output_dir)/B;A.grading_analysis_dir=A.project_output_dir/'temp_analysis';A.llm_model=llm_model;A.emb_model=emb_model;A.decomposition_mode=C if isinstance(C,DecompositionMode)else DecompositionMode(C);A.project_output_dir.mkdir(parents=_B,exist_ok=_B);A.grading_analysis_dir.mkdir(parents=_B,exist_ok=_B)
		if debug:RichLog.set_level(logging.DEBUG)
		if F:
			D=A.project_output_dir/F
			try:RichLog.add_file_handler(str(D),overwrite=_B);RichLog.info(f"[NL2TestActor:{A.project_name}] Writing logs to file: {D}")
			except Exception as H:RichLog.warn(f"[NL2TestActor:{A.project_name}] Failed to add log file handler at {D}: {H}")
		init_config(project_name=A.project_name,base_project_dir=str(A.project_root),project_output_dir=str(A.project_output_dir),llm_provider=cast(Provider,llm_provider),llm_model=A.llm_model,emb_provider=cast(Provider,emb_provider),emb_model=cast(str,A.emb_model),llm_api_url=cast(str,llm_api_url),emb_api_url=cast(str,emb_api_url),llm_api_key=cast(str,os.getenv('LLM_API_KEY')),emb_api_key=cast(str,os.getenv('EMB_API_KEY')),localization_max_iters=localization_max_iters or 40,composition_max_iters=composition_max_iters or 30,supervisor_max_iters=supervisor_max_iters or 10,can_parallel_tool=can_parallel_tool,use_stored_index=use_stored_index,configure_reasoning=configure_reasoning,reasoning_effort=reasoning_effort,exclude_reasoning=exclude_reasoning,max_tokens=max_tokens,openrouter_ignore_providers=openrouter_ignore_providers);A.analysis=CLDK(language='java').analysis(project_path=str(A.project_root),analysis_backend_path=_A,analysis_level=AnalysisLevel.symbol_table,analysis_json_path=A.analysis_dir,eager=_C);A.pipeline=NL2TestPipeline(A.analysis,project_root=A.project_root,analysis_dir=A.analysis_dir,grading_analysis_dir=A.grading_analysis_dir,decomposition_mode=A.decomposition_mode);A.compilation_failed=_C;(A.compilation_failure_payload):dict[str,Any]|_A=_A;A.save_localized_scenarios=save_localized_scenarios;A._ensure_clean_submodule();E=A.pipeline.run_project_compilation()
		if E:G=sorted({A.file for A in E});I=[ErrorFormatter.format_compilation_error(A)for A in E];A.compilation_failed=_B;A.compilation_failure_payload={_D:_C,'project_compilation_failed':_B,_E:'ProjectCompilationError','error':'Project failed to compile before NL2Test run; skipping project.','files_with_errors':G,'error_details':I,'project_name':A.project_name};RichLog.error(f"[NL2TestActor:{A.project_name}] Project failed to compile before NL2Test run. Files with errors: {G}");return
		A.pipeline.run_preprocessing(exclude_test_dirs=exclude_test_dirs)
	def _ensure_clean_submodule(A)->_A:
		if GitUtilities.has_working_tree_changes(A.project_root):RichLog.warn(f"[NL2TestActor:{A.project_name}] Local changes detected in {A.project_root}; resetting.");GitUtilities.reset_submodule(A.project_root)
		if GitUtilities.has_working_tree_changes(A.project_root):raise RuntimeError(f"[NL2TestActor:{A.project_name}] Submodule {A.project_root} still has local changes after reset.")
	def run_nl2test_one(A,input_payload:dict[str,Any])->dict[str,Any]:
		'Run NL2Test for a single input payload.\n\n        Returns a dict with either {success: True, result: NL2TestEval}\n        or {success: False, error: str, error_type: str, traceback: str, input: dict}.\n        ';B=input_payload
		if A.compilation_failed and A.compilation_failure_payload:return A.compilation_failure_payload
		A._ensure_clean_submodule()
		try:
			G=NL2TestInput(**B);C=A.pipeline.run_nl2test(G);E:dict[str,Any]={_D:_B,'result':C.eval}
			if A.save_localized_scenarios and C.localized_scenario is not _A:E['localized_scenario']=C.localized_scenario.model_dump(mode='json')
			return E
		except Exception as D:F=traceback.format_exc();RichLog.error(f"[NL2TestActor:{A.project_name}] Failed for input id={B.get('id')} ({B.get(_F)}::{B.get(_G)}): {D}\n{F}");return{_D:_C,'error':str(D),_E:type(D).__name__,'traceback':F,'input':B}
	def run_nl2test_batch(A,input_payloads:list[dict[str,Any]])->list[dict[str,Any]]:
		'Process a list of inputs sequentially within this actor.\n\n        Ray handles parallelism across actors; this method keeps per-actor\n        behavior simple and logs basic progress.\n        '
		if A.compilation_failed and A.compilation_failure_payload:return[A.compilation_failure_payload]
		C:list[dict[str,Any]]=[]
		for B in tqdm(input_payloads,desc=f"Running NL2TestActor:{A.project_name}",unit='task'):RichLog.debug(f"[NL2TestActor:{A.project_name}] Running {B.get(_F)}::{B.get(_G)} (id={B.get('id')})");C.append(A.run_nl2test_one(B))
		return C