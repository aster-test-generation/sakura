from __future__ import annotations
_B=False
_A=None
import os
from pathlib import Path
from typing import Any
import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from sakura.test2nl.model.models import AbstractionLevel
from sakura.test2nl.pipeline import Pipeline as Test2NLPipeline
from sakura.utils.analysis.java_analyzer import CommonAnalysis
from sakura.utils.config import init_config
from sakura.utils.llm.model import Provider
from sakura.utils.models import Method
@ray.remote
class Test2NLActor:
	'\n    Ray actor that initializes project-scoped analysis and Test2NL pipeline once.\n    '
	def __init__(A,*,project_name:str,analysis_root_dir:str,output_dir:str,llm_model:str,llm_provider:Provider,llm_api_url:str|_A=_A,base_project_dir:str|_A=_A)->_A:C=base_project_dir;B=project_name;A.project_name=B;A.analysis_dir=Path(analysis_root_dir)/B;A.output_dir=Path(output_dir);A.llm_model=llm_model;A.base_project_dir=Path(C)if C is not _A else Path('');A.output_dir.mkdir(parents=True,exist_ok=True);init_config(project_name=A.project_name,base_project_dir=str(A.base_project_dir),project_output_dir=str(A.output_dir),llm_provider=llm_provider,llm_model=A.llm_model,emb_provider=_A,emb_model=_A,llm_api_url=llm_api_url,llm_api_key=os.getenv('LLM_API_KEY'),emb_api_key=_A,localization_max_iters=20);A.analysis=CLDK(language='java').analysis(project_path='',analysis_backend_path=_A,analysis_level=AnalysisLevel.symbol_table,analysis_json_path=A.analysis_dir,eager=_B);A.pipeline=Test2NLPipeline(A.analysis,A.project_name,A.output_dir)
	def generate_descriptions_one(F,input_payload:dict[str,Any])->dict[str,Any]:
		'\n        Generate a description for a single test method.\n        Expects payload with: qualified_class_name, method_signature, id (optional).\n        ';N='json';J='input';I='error';E='success';A=input_payload
		try:
			C=A['qualified_class_name'];B=A['method_signature'];K=B
			if not F.analysis.get_method(C,B):
				L=CommonAnalysis.simplify_method_signature(B)
				if F.analysis.get_method(C,L):K=L
				else:return{E:_B,I:f"Method {B} not found in class {C}",J:A}
			O=Method(qualified_class_name=C,method_signature=K);P=int(A.get('id',0));D=A.get('abstraction_level')
			if D is _A:raise ValueError("Missing required 'abstraction_level' in payload")
			if isinstance(D,AbstractionLevel):M=D
			else:M=AbstractionLevel(str(D).lower())
			G,H=F.pipeline.run_description_of_method(select_method=O,id=P,abstraction=M)
			if G is _A or H is _A:return{E:_B,I:'No description generated',J:A}
			G.method_signature=B;H.method_signature=B;return{E:True,'entry':G.model_dump(mode=N),'description':H.model_dump(mode=N)}
		except Exception as Q:return{E:_B,I:str(Q),J:A}
	def generate_descriptions_batch(B,input_payloads:list[dict[str,Any]])->list[dict[str,Any]]:
		'Sequentially process a batch of methods within this actor.';A:list[dict[str,Any]]=[]
		for C in input_payloads:A.append(B.generate_descriptions_one(C))
		return A