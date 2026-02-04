from __future__ import annotations
_A=None
import os
from pathlib import Path
from typing import Any
import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from sakura.nl2test import Pipeline as NL2TestPipeline
from sakura.nl2test.models import NL2TestInput
from sakura.nl2test.models.decomposition import DecompositionMode
from sakura.utils.analysis import AppJavaAnalysis,CommonAnalysis
from sakura.utils.config import init_config
from sakura.utils.llm.model import Provider
@ray.remote
class LocalizationActor:
	'\n    Ray actor that initializes config and CLDK analysis once per project.\n    '
	def __init__(A,*,project_name:str,base_project_dir:str,output_dir:str,llm_model:str,emb_model:str|_A,llm_provider:Provider,llm_api_url:str|_A,emb_provider:Provider,emb_api_url:str|_A,decomposition_mode:str|DecompositionMode,localization_max_iters:int)->_A:D=decomposition_mode;C=project_name;A.project_name=C;A.project_root=Path(base_project_dir)/C;A.project_output_dir=Path(output_dir)/C;A.llm_model=llm_model;A.emb_model=emb_model;A.decomposition_mode=D if isinstance(D,DecompositionMode)else DecompositionMode(D);A.localization_max_iters=localization_max_iters;A.project_output_dir.mkdir(parents=True,exist_ok=True);init_config(project_name=A.project_name,base_project_dir=str(A.project_root),project_output_dir=str(A.project_output_dir),llm_provider=llm_provider,llm_model=A.llm_model,emb_provider=emb_provider,emb_model=A.emb_model,llm_api_url=llm_api_url,emb_api_url=emb_api_url,llm_api_key=os.getenv('LLM_API_KEY'),emb_api_key=os.getenv('EMB_API_KEY'),localization_max_iters=A.localization_max_iters or 20);B=CLDK(language='java').analysis(project_path=A.project_root,analysis_backend_path=_A,analysis_level=AnalysisLevel.symbol_table,analysis_json_path=A.project_output_dir,eager=False);E=CommonAnalysis(B);I,F,G=E.categorize_classes();H=AppJavaAnalysis(application_classes=F,project_dir=B.project_dir,source_code=B.source_code,analysis_backend_path=B.analysis_backend_path,analysis_json_path=B.analysis_json_path,analysis_level=B.analysis_level,target_files=B.target_files,eager_analysis=B.eager_analysis);A.pipeline=NL2TestPipeline(H,project_root=A.project_root,analysis_dir=A.project_output_dir,decomposition_mode=A.decomposition_mode,application_classes=F,test_utility_classes=G,common_analysis=E,grading_analysis=B);A.pipeline.run_preprocessing()
	def localize_one(C,input_payload:dict[str,Any])->dict[str,Any]:
		'Run localization evaluation for a single NL2Test input payload.';B='success';A=input_payload
		try:D=NL2TestInput(**A);E=C.pipeline.run_localization_evaluation_pipeline(D);return{B:True,'output':E.model_dump(mode='json')}
		except Exception as F:return{B:False,'error':str(F),'input':A}
	def localize_batch(B,input_payloads:list[dict[str,Any]])->list[dict[str,Any]]:
		'Run localization evaluation for a batch of payloads sequentially in this actor.';A:list[dict[str,Any]]=[]
		for C in input_payloads:A.append(B.localize_one(C))
		return A