_F='erroneous_count'
_E='erroneous_tests'
_D='project_name'
_C='utf-8'
_B=True
_A=None
import json,os,shutil
from pathlib import Path
from typing import Any,Dict,List
import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from tqdm import tqdm
from sakura.dataset_creation.model import NL2TestDataset,Test
from sakura.utils.analysis.java_analyzer import CommonAnalysis
ROOT_DIR=Path(__file__).resolve().parent.parent.parent.parent
INPUT_FILE_NAME='nl2test.json'
OUTPUT_FILE_NAME='erroneous_tests.json'
RESOURCES_DIR='resources'
TESTS_DIR='filtered_bucketed_tests'
ANALYSIS_DIR='analysis'
DATASETS_DIR='datasets'
ERRONEOUS_TESTS_DIR=_E
def get_all_tests(dataset:NL2TestDataset)->List[Test]:'Collect all tests from all buckets in the dataset.';A=dataset;return A.tests_with_one_focal_methods+A.tests_with_two_focal_methods+A.tests_with_more_than_two_to_five_focal_methods+A.tests_with_more_than_five_to_ten_focal_methods+A.tests_with_more_than_ten_focal_methods
@ray.remote
def _verify_project(project_name:str,tests_file:str,analysis_path:str,project_path:str,output_dir:str)->Dict[str,Any]|_A:
	'\n    Verify all tests in a project can be found via CLDK analysis.\n\n    Returns verification result dict or None if processing fails.\n    ';C=project_name
	with open(tests_file,'r',encoding=_C)as D:H=json.load(D)
	I=NL2TestDataset(**H)
	try:F=CLDK(language='java').analysis(project_path=project_path,analysis_backend_path=_A,analysis_level=AnalysisLevel.symbol_table,analysis_json_path=analysis_path,eager=_B)
	except Exception:return
	B:List[Dict[str,str]]=[];J=get_all_tests(I)
	for A in J:
		E=F.get_method(A.qualified_class_name,A.method_signature)
		if not E:K=CommonAnalysis.simplify_method_signature(A.method_signature);E=F.get_method(A.qualified_class_name,K)
		if not E:B.append({'qualified_class_name':A.qualified_class_name,'method_signature':A.method_signature})
	if B:
		G=Path(output_dir)/C;G.mkdir(parents=_B,exist_ok=_B);L=G/OUTPUT_FILE_NAME;M={_D:C,'erroneous_test_count':len(B),_E:B}
		with open(L,'w',encoding=_C)as D:json.dump(M,D,indent=2)
	return{_D:C,_F:len(B)}
def _create_summary(output_dir:Path,results:List[Dict[str,Any]|_A])->_A:
	'Create a summary.json with verification results per project.';G='total_erroneous_tests';F='projects_with_errors';E='total_projects';B='projects';A:Dict[str,Any]={E:0,F:0,G:0,B:{}}
	for C in results:
		if C is _A:continue
		A[E]+=1;D=C[_F]
		if D>0:A[F]+=1;A[G]+=D;A[B][C[_D]]=D
	A[B]=dict(sorted(A[B].items()));H=output_dir/'summary.json'
	with open(H,'w',encoding=_C)as I:json.dump(A,I,indent=2)
	print('\nVerification complete:');print(f"  Total projects: {A[E]}");print(f"  Projects with errors: {A[F]}");print(f"  Total erroneous tests: {A[G]}");print(f"  Summary saved to: {H}")
def process_projects(tests_dir:Path,analysis_dir:Path,datasets_dir:Path,output_dir:Path)->_A:
	'\n    Process all projects in tests_dir in parallel, verify tests against analysis,\n    and output erroneous tests.\n    ';D=tests_dir;B=output_dir
	if B.exists():shutil.rmtree(B)
	B.mkdir(parents=_B,exist_ok=_B);C=[]
	for A in sorted(os.listdir(D)):
		if A.startswith('.')or A.startswith('__'):continue
		E=D/A
		if not E.is_dir():continue
		F=E/INPUT_FILE_NAME
		if not F.exists():print(f"Skipping {A}: no {INPUT_FILE_NAME} in tests directory");continue
		G=analysis_dir/A
		if not G.exists():print(f"Skipping {A}: no analysis directory found");continue
		H=datasets_dir/A
		if not H.exists():print(f"Skipping {A}: no source project found in datasets");continue
		C.append(_verify_project.remote(A,str(F),str(G),str(H),str(B)))
	I:List[Dict[str,Any]|_A]=[]
	with tqdm(total=len(C),desc='Verifying projects...')as K:
		while C:J,C=ray.wait(C,num_returns=1);L=ray.get(J);I.extend(L);K.update(len(J))
	_create_summary(B,I)
def main():A=ROOT_DIR/RESOURCES_DIR;B=A/TESTS_DIR;C=A/ANALYSIS_DIR;D=A/DATASETS_DIR;E=A/ERRONEOUS_TESTS_DIR;process_projects(B,C,D,E)
if __name__=='__main__':main()