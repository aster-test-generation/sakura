from __future__ import annotations
_A=None
import os
from pathlib import Path
from typing import Any
import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from nltest.nl2test import Pipeline as NL2TestPipeline
from nltest.nl2test.models import NL2TestInput
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.utils.config import init_config
from nltest.utils.llm.model import Provider
@ray.remote
class LocalizationActor:
	'\n    Ray actor that initializes config and CLDK analysis once per project.\n    '
	def __init__(A,*,project_name:str,base_project_dir:str,output_dir:str,llm_model:str,emb_model:str|_A,llm_provider:Provider,llm_api_url:str|_A,emb_provider:Provider,emb_api_url:str|_A,decomposition_mode:str|DecompositionMode,localization_max_iters:int)->_A:C=decomposition_mode;B=project_name;A.project_name=B;A.project_root=Path(base_project_dir)/B;A.project_output_dir=Path(output_dir)/B;A.llm_model=llm_model;A.emb_model=emb_model;A.decomposition_mode=C if isinstance(C,DecompositionMode)else DecompositionMode(C);A.localization_max_iters=localization_max_iters;A.project_output_dir.mkdir(parents=True,exist_ok=True);init_config(project_name=A.project_name,base_project_dir=str(A.project_root),project_output_dir=str(A.project_output_dir),llm_provider=llm_provider,llm_model=A.llm_model,emb_provider=emb_provider,emb_model=A.emb_model,llm_api_url=llm_api_url,emb_api_url=emb_api_url,llm_api_key=os.getenv('LLM_API_KEY'),emb_api_key=os.getenv('EMB_API_KEY'),localization_max_iters=A.localization_max_iters or 20);A.analysis=CLDK(language='java').analysis(project_path=A.project_root,analysis_backend_path=_A,analysis_level=AnalysisLevel.symbol_table,analysis_json_path=A.project_output_dir,eager=False);A.pipeline=NL2TestPipeline(A.analysis,project_root=A.project_root,analysis_dir=A.project_output_dir,decomposition_mode=A.decomposition_mode);A.pipeline.run_preprocessing()
	def localize_one(C,input_payload:dict[str,Any])->dict[str,Any]:
		'Run localization evaluation for a single NL2Test input payload.';B='success';A=input_payload
		try:D=NL2TestInput(**A);E=C.pipeline.run_localization_evaluation_pipeline(D);return{B:True,'output':E.model_dump(mode='json')}
		except Exception as F:return{B:False,'error':str(F),'input':A}
	def localize_batch(B,input_payloads:list[dict[str,Any]])->list[dict[str,Any]]:
		'Run localization evaluation for a batch of payloads sequentially in this actor.';A:list[dict[str,Any]]=[]
		for C in input_payloads:A.append(B.localize_one(C))
		return A