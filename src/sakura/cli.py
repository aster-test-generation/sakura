_T='medium'
_S='method_signature'
_R='qualified_class_name'
_Q='analysis.json'
_P='Either --llm-provider or --llm-api-url must be provided. Provider only guides the API URL default.'
_O='OpenAI-compatible base URL for the LLM API (must support the OpenAI API format).'
_N='LLM provider (guides default API URL). One of: openrouter, vllm, ollama, openai, gcp. Either this or --llm-api-url must be provided.'
_M='Maximum number of projects to process concurrently.'
_L='nl2test_localized_scenarios.json'
_K='nl2test_evaluation_results.json'
_J='error'
_I='success'
_H='csv'
_G='nl2test_failures.json'
_F='id'
_E='append'
_D='json'
_C=True
_B=False
_A=None
import logging,shutil
from collections import deque
from pathlib import Path
import ray,typer
from dotenv import load_dotenv
from typing_extensions import Annotated
from sakura.dataset_creation.model import NL2TestDataset
from sakura.dataset_creation.model import Test as DatasetTest
from sakura.nl2test.models import NL2TestEval,NL2TestInput
from sakura.nl2test.models.decomposition import DecompositionMode
from sakura.ray_utils.nl2test_actor import NL2TestActor
from sakura.ray_utils.test2nl_actor import Test2NLActor
from sakura.test2nl.model.models import AbstractionLevel,Test2NLEntry
from sakura.utils.file_io.structured_data_manager import StructuredDataManager
from sakura.utils.llm.model import Provider
from sakura.utils.models import NL2TestFailure
from sakura.utils.pretty.color_logger import RichLog
from sakura.utils.vcs.git_utils import GitUtilities
app=typer.Typer(help='Sakura',pretty_exceptions_enable=_B,pretty_exceptions_show_locals=_B,add_completion=_B)
load_dotenv()
IGNORED_DIRS={'__pycache__','.git','.idea','.vscode'}
NL2TEST_DEBUG=_B
@app.callback()
def main()->_A:0
def _load_nl2_inputs_by_project_from_csv(test2nl_file:str|Path,max_entries:int,num_proj_parallel:int,target_ids:list[int]|_A=_A)->dict[str,list[NL2TestInput]]:
	'Load Test2NL CSV (explicit file path) and convert to NL2TestInput grouped by project.';H=test2nl_file;G=target_ids;F=max_entries
	if H is _A:raise Exception('Parameter --test2nl-file is required and was not provided.')
	C=Path(H)
	if not C.exists():raise Exception(f"CSV file {C} does not exist.")
	if not C.is_file():raise Exception(f"Expected --test2nl-file to point to a CSV file, but {C} is not a file.")
	if C.suffix.lower()!='.csv':raise Exception(f"Expected --test2nl-file to have a .csv extension, but got {C.name}.")
	RichLog.info(f"Loading Test2NL entries from {C}");I=StructuredDataManager(C.parent);A=I.load(C.name,Test2NLEntry,format=_H);J=len(A)
	if NL2TEST_DEBUG:
		E=Path(__file__).parent.parent.parent/'resources/test2nl/filtered_dataset/spanning_subset_20.csv'
		if not E.exists():raise Exception(f"Debug CSV file not found: {E}")
		K=StructuredDataManager(E.parent);A=K.load(E.name,Test2NLEntry,format=_H);RichLog.info(f"NL2TEST_DEBUG enabled: loaded {len(A)} entries from {E}")
	if G:L=set(G);A=[A for A in A if A.id in L];RichLog.info(f"Filtered to {len(A)} entries matching target_ids={G}")
	if not NL2TEST_DEBUG and F>0:A=A[:F];RichLog.info(f"Processing subset of {len(A)} entries (max_entries={F}, total_entries={J})")
	A.sort(key=lambda entry:(entry.qualified_class_name,entry.method_signature));RichLog.info(f"Loaded and sorted {len(A)} Test2NL entries by class-method pairs");D:dict[str,list[NL2TestInput]]={}
	for B in A:
		M=NL2TestInput(description=B.description,project_name=B.project_name,qualified_class_name=B.qualified_class_name,method_signature=B.method_signature,abstraction_level=B.abstraction_level.value,is_bdd=B.is_bdd,id=B.id)
		if B.project_name not in D:D[B.project_name]=[]
		D[B.project_name].append(M)
	RichLog.info(f"Organized inputs by {len(D)} projects: {list(D.keys())}");return D
def _clear_nl2test_output_artifacts(output_dir:Path,reset_evaluation_results:bool)->_A:
	B=output_dir
	if not B.exists():return
	C=B/'nl2test.log'
	if C.exists():C.unlink()
	for A in B.iterdir():
		if not A.is_dir():continue
		if reset_evaluation_results:
			D=A/_K
			if D.exists():D.unlink()
			E=A/_G
			if E.exists():E.unlink()
			F=A/_L
			if F.exists():F.unlink()
			G=A/'code_iteration'
			if G.exists():shutil.rmtree(G)
		H=A/'temp_analysis'
		if H.exists():shutil.rmtree(H)
		for I in A.glob('*.log'):I.unlink()
def _create_failure(payload:dict,error:str,error_type:str,traceback:str|_A=_A)->NL2TestFailure:return NL2TestFailure(nl2test_input=NL2TestInput(**payload),error=error,error_type=error_type,traceback=traceback)
@app.command()
def generate_descriptions(analysis_dir:Annotated[str,typer.Option(help='Path to the directory containing all project analysis directories (each with an analysis.json).',show_default=_B)],output_dir:Annotated[str,typer.Option(help='Path to the output directory where test2nl.csv and descriptions.json are saved.',show_default=_B)],llm_model:Annotated[str,typer.Option(help='LLM model ID to use for generating Test2NL descriptions.',show_default=_B)],organized_methods_dir:Annotated[str,typer.Option(help='Path to the directory containing per-project folders of filtered methods (each holding the filtered JSON file).',show_default=_B)],organized_methods_file_name:Annotated[str,typer.Option(help='Name of the JSON file containing filtered methods for each project within organized_methods_dir.',show_default=_B)]='nl2test.json',clear_dataset:Annotated[bool,typer.Option(help='Whether to clear existing Test2NL data at the output directory before appending.',show_default=_B)]=_C,max_methods:Annotated[int,typer.Option(help='Maximum number of test methods to process across all projects (0 for unlimited). Note: generates 3x entries due to three abstraction levels per method.',show_default=_B)]=0,num_proj_parallel:Annotated[int,typer.Option(help=_M,show_default=_C)]=2,per_proj_concurrency:Annotated[int,typer.Option(help='Maximum concurrent generate_descriptions_one calls per project.',show_default=_C)]=2,max_inflight:Annotated[int,typer.Option(help='Global cap on in-flight tasks across all projects (0 uses 2 * num_proj_parallel * per_proj_concurrency).',show_default=_C)]=0,exclude_groups:Annotated[list[str],typer.Option(help='Dataset group names to exclude (repeat the option to exclude multiple).',show_default=_C)]=[],llm_provider:Annotated[str|_A,typer.Option(help=_N,show_default=_B)]=_A,llm_api_url:Annotated[str|_A,typer.Option(help=_O,show_default=_B)]=_A):
	'\n    Generate Test2NL descriptions for methods that passed the filtering pipeline.\n\n    The organized_methods_dir must contain per-project folders with a JSON filter\n    that enumerates the methods to process. Only methods listed in that file are\n    used when producing descriptions.\n    ';w='total';v='actor';u='test2nl.csv';t='descriptions.json';i=llm_api_url;h=num_proj_parallel;g=clear_dataset;f='produced';X=exclude_groups;W=max_inflight;V=per_proj_concurrency;U='pending';N=max_methods;M=output_dir;L='inflight';F=llm_provider;D=organized_methods_file_name
	if g is _B:raise NotImplementedError('Behavior for retrieving the max ID for continuing dataset appends is not implemented.')
	if F is _A and i is _A:raise Exception(_P)
	if F is not _A:
		try:F=Provider(F.strip().lower())
		except Exception:raise Exception(f"Invalid --llm-provider: {F}. Must be one of {[A.value for A in Provider]}")
	M=Path(M);I=Path(analysis_dir);G=Path(organized_methods_dir);D=D.strip()
	if not D:raise Exception('Parameter --organized-methods-file-name cannot be empty.')
	if Path(D).suffix.lower()!='.json':raise Exception('Parameter --organized-methods-file-name must point to a JSON file name.')
	if not(G.exists()and G.is_dir()):raise Exception(f"Organized methods directory {G} does not exist or is not a directory.")
	if not(I.exists()and I.is_dir()):raise Exception(f"Analysis directory {I} does not exist or is not a directory.")
	O=[A for A in sorted(G.iterdir())if A.is_dir()and A.name not in IGNORED_DIRS and not A.name.startswith('.')and(A/D).exists()]
	if not O:RichLog.warn(f"No project directories with {D} found under {G}");return
	Y=StructuredDataManager(M)
	if g:RichLog.info('Clearing the existing Test2NL dataset at the output directory.');x=[t,u];Y.delete_many(x)
	RichLog.info(f"Found {len(O)} project(s) with {D}: {[A.name for A in O]}");J:dict[str,list[dict]]={};j=0
	for k in O:
		A=k.name;P=Path(I)/A;l=P/_Q
		if not(P.exists()and P.is_dir()):RichLog.warn(f"Skipping {A}: missing analysis directory {P}");continue
		if not l.exists():RichLog.warn(f"Skipping {A}: missing analysis.json at {l}");continue
		m=k/D
		try:y=NL2TestDataset.model_validate_json(m.read_text('utf-8'))
		except Exception as H:RichLog.warn(f"Skipping {A}: failed to load dataset from {m} ({H})");continue
		K=['tests_with_one_focal_methods','tests_with_two_focal_methods','tests_with_more_than_two_to_five_focal_methods','tests_with_more_than_five_to_ten_focal_methods','tests_with_more_than_ten_focal_methods']
		if X:
			n=set(K);o=[A for A in X if A not in n]
			if o:RichLog.warn(f"Unknown group(s) in exclude_groups: {o}. Valid groups: {sorted(n)}")
			K=[A for A in K if A not in set(X)]
		if not K:RichLog.warn(f"All dataset groups were excluded for project {A}; skipping.");J[A]=[];continue
		E:list[dict]=[]
		for z in K:
			A0:list[DatasetTest]=getattr(y,z,[])or[]
			for p in A0:E.append({_R:p.qualified_class_name,_S:p.method_signature,_F:0})
		if N>0:
			Q=N-j
			if Q<=0:RichLog.info(f"Reached global max_methods limit ({N}). Skipping remaining projects.");break
			if len(E)>Q:RichLog.info(f"Limiting tests for {A} to first {Q} out of {len(E)} due to max_methods={N}");E=E[:Q]
		J[A]=E;j+=len(E)
	if not J:RichLog.warn('No valid project payloads to process. Exiting.');return
	try:
		if not ray.is_initialized():ray.init()
	except Exception as H:raise RuntimeError(f"Failed to initialize Ray: {H}")from H
	Z=0;R=deque(J.keys());B:dict[str,dict]={};q:dict[ray.ObjectRef,str]={};C:set[ray.ObjectRef]=set();r=W if W and W>0 else 2*max(1,int(h))*max(1,int(V))
	def a()->_A:
		while len(B)<max(1,int(h))and R:
			A=R.popleft();D='='*60;RichLog.info(f"\n{D}");RichLog.info(f"Starting Test2NL actor for project: {A}");RichLog.info(str(D));H=Test2NLActor.options(max_concurrency=max(1,int(V))).remote(project_name=A,analysis_root_dir=str(I),output_dir=str(M),llm_model=llm_model,llm_provider=F,llm_api_url=i,base_project_dir=str(G/A));K=list(J.get(A,[]));C:list[dict]=[]
			for N in K:
				for O in AbstractionLevel:E=dict(N);E['abstraction_level']=O.value;C.append(E)
			B[A]={v:H,U:deque(C),L:set(),w:len(C),f:0}
			if len(C)==0:RichLog.warn(f"No inputs for project {A}; will finalize immediately.")
	def s()->_A:
		nonlocal C
		if len(C)>=r:return
		D=r-len(C)
		if D<=0:return
		F=max(1,int(V))
		for(G,A)in list(B.items()):
			if D<=0:break
			while D>0 and len(A[L])<F and A[U]:H=A[U].popleft();E=A[v].generate_descriptions_one.remote(H);A[L].add(E);C.add(E);q[E]=G;D-=1
	def b(pname:str)->_A:
		C=pname;A=B.get(C)
		if A is _A:return
		if A[U]or A[L]:return
		RichLog.info(f"Project {C} completed: total_methods={A[w]}, produced={A[f]}");B.pop(C,_A)
	a();s()
	while B or R or C:
		a();s()
		if not C:
			for A1 in list(B.keys()):b(A1)
			a()
			if not C and not B and not R:break
			continue
		A2,_=ray.wait(list(C),num_returns=1)
		for S in A2:
			A=q.pop(S,_A);C.discard(S)
			if A is _A:continue
			c=B.get(A)
			if c is _A:continue
			c[L].discard(S)
			try:T=ray.get(S)
			except Exception as H:RichLog.error(f"[{A}] Ray task failed: {H}");b(A);continue
			if T.get(_I):
				d=T.get('entry');e=T.get('description')
				if d and e:d[_F]=Z;e[_F]=Z;Y.save(t,[e],format=_D,mode=_E);Y.save(u,[d],format=_H,mode=_E);Z+=1;c[f]+=1
				else:RichLog.warn(f"[{A}] Received success but missing payloads; skipping.")
			else:A3=T.get(_J,'Unknown error');RichLog.error(f"[{A}] Failed to generate description: {A3}")
			b(A)
	try:
		if ray.is_initialized():ray.shutdown()
	except Exception:pass
@app.command()
def run_nl2test(base_project_dir:Annotated[str,typer.Option(help='Path to the base directory containing all project directories.',show_default=_B)],base_analysis_dir:Annotated[str,typer.Option(help='Path to the base directory containing per-project analysis.json directories.',show_default=_B)],output_dir:Annotated[str,typer.Option(help='Path to the output directory for saving NL2Test generation results.',show_default=_B)],reset_evaluation_results:Annotated[bool,typer.Option(help='Whether to remove existing NL2Test evaluation results before running.',show_default=_C)]=_C,use_stored_index:Annotated[bool,typer.Option(help='Reuse cached FAISS indexes when available instead of rebuilding them.',show_default=_C)]=_C,test2nl_file:Annotated[str,typer.Option(help='Path to the Test2NL CSV file to use as inputs (e.g., /path/to/test2nl.csv).',show_default=_B)]=_A,llm_model:Annotated[str,typer.Option(help='LLM model to use for NL2Test generation.',show_default=_B)]='mistralai/devstral-small',can_parallel_tool:Annotated[bool,typer.Option(help='Whether the LLM client may issue parallel tool calls.',show_default=_C)]=_C,emb_model:Annotated[str,typer.Option(help='Embedding model to use for vector search.',show_default=_B)]='nomic-embed-text:v1.5',decomposition_mode:Annotated[str,typer.Option(help="Decomposition mode (must be 'gherkin' for now).",show_default=_C)]='gherkin',supervisor_max_iters:Annotated[int,typer.Option(help='Maximum iterations for the supervisor agent.',show_default=_C)]=10,localization_max_iters:Annotated[int,typer.Option(help='Maximum iterations for the localization agent.',show_default=_C)]=40,composition_max_iters:Annotated[int,typer.Option(help='Maximum iterations for the composition agent.',show_default=_C)]=30,num_proj_parallel:Annotated[int,typer.Option(help=_M,show_default=_C)]=2,max_inflight:Annotated[int,typer.Option(help='Global cap on in-flight project tasks (0 uses num_proj_parallel).',show_default=_C)]=0,max_entries:Annotated[int,typer.Option(help='Maximum number of Test2NL entries to process (0 for all).',show_default=_B)]=0,target_ids:Annotated[list[int]|_A,typer.Option(help='Only process Test2NL entries with these IDs (omit to process all).',show_default=_B)]=_A,llm_provider:Annotated[str|_A,typer.Option(help=_N,show_default=_B)]=_A,llm_api_url:Annotated[str|_A,typer.Option(help=_O,show_default=_B)]=_A,emb_provider:Annotated[str|_A,typer.Option(help='Embedding provider (guides default API URL). One of: vllm, ollama, openai, openrouter, gcp. Either this or --emb-api-url must be provided.',show_default=_B)]=_A,emb_api_url:Annotated[str|_A,typer.Option(help='Base URL for the Embedding API if using an HTTP endpoint.',show_default=_B)]=_A,debug:Annotated[bool,typer.Option(help='Enable debug logging for more verbose output.',show_default=_C)]=_B,log_file:Annotated[str|_A,typer.Option(help='Optional log file name to write under --output-dir.',show_default=_B)]=_A,exclude_test_dirs:Annotated[bool,typer.Option(help='Skip files under Maven test directories when preparing indexes.',show_default=_C)]=_B,configure_reasoning:Annotated[bool,typer.Option(help='Configure reasoning parameters for OpenRouter provider (only applies when using OpenRouter).',show_default=_C)]=_B,reasoning_effort:Annotated[str,typer.Option(help='Reasoning effort level. One of: none, minimal, low, medium, high, xhigh.',show_default=_C)]=_T,exclude_reasoning:Annotated[bool,typer.Option(help='Exclude reasoning content from LLM responses (set to False for MiniMax models to see thinking between tool calls).',show_default=_C)]=_C,max_tokens:Annotated[int,typer.Option(help='Maximum tokens for LLM completion output. Increase for reasoning models (e.g., 32768 or 65536 for MiniMax M2.1).',show_default=_C)]=16384,openrouter_ignore_providers:Annotated[list[str],typer.Option(help='Provider names to exclude when routing through OpenRouter (can be repeated).')]=[],save_localized_scenarios:Annotated[bool,typer.Option(help='Save localized scenarios to a separate JSON file per-project (GHERKIN mode only).',show_default=_C)]=_B,store_code_iteration:Annotated[bool,typer.Option(help='Save each generate_test_code iteration to code_iteration/ directory for debugging.',show_default=_C)]=_B):
	A8='nl2test_input';A7='result';s=save_localized_scenarios;r=openrouter_ignore_providers;q=debug;p=emb_api_url;o=llm_api_url;n=target_ids;m=max_entries;l=emb_model;k=llm_model;j='localized_scenario';a=log_file;Z=num_proj_parallel;T=max_inflight;O=reasoning_effort;N=decomposition_mode;J=emb_provider;I=llm_provider;F=base_analysis_dir;D=base_project_dir;B=output_dir
	try:N=DecompositionMode(N.strip().lower())
	except Exception:raise Exception(f"Invalid --decomposition-mode: {N}. Must be one of {[A.value for A in DecompositionMode]}")
	if N!=DecompositionMode.GHERKIN:raise Exception("Only 'gherkin' decomposition is supported for run_nl2test at the moment.")
	t={'none','minimal','low',_T,'high','xhigh'};O=O.strip().lower()
	if O not in t:raise Exception(f"Invalid --reasoning-effort: {O}. Must be one of {sorted(t)}")
	if I is _A and o is _A:raise Exception(_P)
	if J is _A and p is _A:raise Exception('Either --emb-provider or --emb-api-url must be provided for embeddings. Provider only guides the API URL default.')
	if I is not _A:
		try:I=Provider(I.strip().lower())
		except Exception:raise Exception(f"Invalid --llm-provider: {I}. Must be one of {[A.value for A in Provider]}")
	if J is not _A:
		try:J=Provider(J.strip().lower())
		except Exception:raise Exception(f"Invalid --emb-provider: {J}. Must be one of {[A.value for A in Provider]}")
	D=Path(D).expanduser().resolve()
	if not(D.exists()and D.is_dir()):raise Exception(f"Base project directory {D} does not exist.")
	F=Path(F).expanduser().resolve()
	if not(F.exists()and F.is_dir()):raise Exception(f"Base analysis directory {F} does not exist.")
	B=Path(B).expanduser().resolve()
	if B.exists()and not B.is_dir():raise Exception(f"Output path {B} is not a directory.")
	if not B.exists():B.mkdir(parents=_C,exist_ok=_C)
	if n and m>0:raise ValueError('Cannot specify both --target-ids and --max-entries. Use --target-ids to filter specific entries, or --max-entries to limit the count.')
	RichLog.info(f"Resetting submodules under {D}");GitUtilities.reset_submodules_in_dir(D);P=_load_nl2_inputs_by_project_from_csv(test2nl_file,m,Z,n);_clear_nl2test_output_artifacts(B,reset_evaluation_results);u:str|_A=_A
	if q:RichLog.set_level(logging.DEBUG);RichLog.debug('Debug logging enabled.')
	if a:
		v=Path(a).name;u=v
		try:w=B/v;RichLog.add_file_handler(str(w),overwrite=_C);RichLog.info(f"Writing logs to file: {w}")
		except Exception as G:RichLog.warn(f"Failed to add file handler at {a}: {G}")
	RichLog.info(f"NL2Test run configuration: llm_model={k}, emb_model={l}, provider={I}, emb_provider={J}, projects_dir={D}, analysis_dir={F}, out={B}, max_inflight={T or Z}");M=deque()
	for A in P.keys():
		U=D/A;A9=F/A;b=A9/_Q
		if not(U.exists()and U.is_dir()):RichLog.error(f"Project directory {U} does not exist.");raise Exception(f"Project directory {U} does not exist.")
		if not b.exists():RichLog.error(f"Missing analysis.json for {A} at {b}.");raise Exception(f"Missing analysis.json for {A} at {b}.")
		M.append(A)
	if not M:RichLog.error('No valid projects to process. Exiting.');return
	try:
		if not ray.is_initialized():ray.init()
	except Exception as G:raise RuntimeError(f"Failed to initialize Ray: {G}")from G
	AA=_K;x:dict[str,StructuredDataManager]={}
	def c(project_name:str)->StructuredDataManager:
		C=project_name;A=x.get(C)
		if A is _A:A=StructuredDataManager(B/C);x[C]=A
		return A
	y=0;V=0;K:set[ray.ObjectRef]=set();z:dict[ray.ObjectRef,str]={};W:dict[str,int]={};AB=T if T and T>0 else max(1,int(Z))
	def Q()->_A:
		while M and len(K)<AB:A=M.popleft();C='='*60;RichLog.info(f"\n{C}");RichLog.info(f"Starting NL2Test actor for project: {A}");RichLog.info(str(C));H=NL2TestActor.options(max_concurrency=1).remote(project_name=A,base_project_dir=str(D),base_analysis_dir=str(F),output_dir=str(B),llm_model=k,can_parallel_tool=can_parallel_tool,emb_model=l,llm_provider=I,llm_api_url=o,emb_provider=J,emb_api_url=p,decomposition_mode=N.value,supervisor_max_iters=supervisor_max_iters,localization_max_iters=localization_max_iters,composition_max_iters=composition_max_iters,use_stored_index=use_stored_index,debug=bool(q),log_file_name=u,exclude_test_dirs=exclude_test_dirs,configure_reasoning=configure_reasoning,reasoning_effort=O,exclude_reasoning=exclude_reasoning,max_tokens=max_tokens,openrouter_ignore_providers=r if r else _A,save_localized_scenarios=s,store_code_iteration=store_code_iteration);E=[A.model_dump(mode=_D)for A in P.get(A,[])];W[A]=len(E);G=H.run_nl2test_batch.remote(E);K.add(G);z[G]=A
	Q()
	while K or M:
		if not K:
			Q()
			if not K and not M:break
		AC,_=ray.wait(list(K),num_returns=1)
		for d in AC:
			A=z.pop(d,_A);K.discard(d)
			if A is _A:continue
			try:R=ray.get(d)
			except Exception as G:
				RichLog.error(f"[{A}] NL2Test batch failed: {G}");V+=W.get(A,0);E=[_create_failure(A.model_dump(mode=_D),str(G),type(G).__name__)for A in P.get(A,[])]
				if E:L=c(A);L.save(_G,E,format=_D,mode=_E)
				Q();continue
			if R and isinstance(R,list):
				X=next((A for A in R if isinstance(A,dict)and A.get('project_compilation_failed')),_A)
				if X:
					A0=X.get(_J);RichLog.error(f"[{A}] {A0 or'Project failed baseline compilation; skipping project.'}");A1=X.get('files_with_errors')or[]
					if A1:RichLog.error(f"[{A}] Files with compilation errors: {A1}")
					A2=X.get('error_details')or[]
					if A2:RichLog.error(f"[{A}] Compilation error details:\n"+'\n'.join(A2))
					V+=W.get(A,0);E=[_create_failure(A.model_dump(mode=_D),A0 or'Project failed baseline compilation','ProjectCompilationError')for A in P.get(A,[])]
					if E:L=c(A);L.save(_G,E,format=_D,mode=_E)
					Q();continue
			e=0;Y=0;f:list[NL2TestEval]=[];E:list[NL2TestFailure]=[]
			for C in R or[]:
				if C.get(_I):
					S=C.get(A7)
					if S is not _A:
						if isinstance(S,dict):S=NL2TestEval(**S)
						f.append(S);e+=1
					else:
						RichLog.warn(f"[{A}] Success reported but result is None");H=C.get('input')or{}
						if H:E.append(_create_failure(H,'Success reported but result is None','EmptyResult'))
						Y+=1
				else:
					A3=C.get(_J);A4=C.get('error_type')or'Exception';H=C.get('input')or{};RichLog.error(f"[{A}] Failed input id={H.get(_F)} {H.get(_R)}::{H.get(_S)} -> {A4}: {A3}");g=C.get('traceback')
					if g:RichLog.debug(g)
					if H:E.append(_create_failure(H,A3 or'',A4,g))
					Y+=1
			L=c(A)
			if f:L.save(AA,f,format=_D,mode=_E)
			if E:L.save(_G,E,format=_D,mode=_E)
			if s:
				h=[]
				for C in R or[]:
					if C.get(_I)and C.get(j):
						i=C.get(A7)
						if isinstance(i,dict):A5=(i.get(A8)or{}).get(_F,-1)
						else:A6=getattr(i,A8,_A);A5=getattr(A6,_F,-1)if A6 else-1
						h.append({'input_id':A5,j:C[j]})
				if h:L.save(_L,h,format=_D,mode=_E)
			y+=e;V+=Y;RichLog.info(f"[{A}] Completed NL2Test: success={e}, failed={Y} (inputs={W.get(A,0)})");Q()
	RichLog.info(f"Overall NL2Test: projects={len(P)}, success={y}, failed={V}")
	try:
		if ray.is_initialized():ray.shutdown()
	except Exception:pass
if __name__=='__main__':app()