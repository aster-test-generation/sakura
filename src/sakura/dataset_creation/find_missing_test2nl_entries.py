_N='tests_with_more_than_ten_focal_methods'
_M='tests_with_more_than_five_to_ten_focal_methods'
_L='tests_with_more_than_two_to_five_focal_methods'
_K='tests_with_two_focal_methods'
_J='tests_with_one_focal_methods'
_I='nl2test.json'
_H='missing_count'
_G=True
_F='incomplete_abstractions'
_E='completely_missing'
_D='project_name'
_C='utf-8'
_B=None
_A='total_tests'
import csv,json,os,shutil
from pathlib import Path
from typing import Any,Dict,List,Set,Tuple
import ray
from tqdm import tqdm
from sakura.dataset_creation.model import NL2TestDataset,Test
ROOT_DIR=Path(__file__).resolve().parent.parent.parent.parent
TEST2NL_DIR='resources/test2nl/filtered_dataset'
TEST2NL_FILE='test2nl.csv'
BUCKETED_DATASET_DIR='resources/filtered_bucketed_tests'
BUCKETED_FILE=_I
MISSING_TESTS_DIR='resources/missing_tests'
OUTPUT_FILE=_I
REQUIRED_ABSTRACTION_LEVELS={'low','medium','high'}
def load_test2nl_entries(csv_path:Path)->Dict[Tuple[str,str,str],Set[str]]:
	'\n    Load Test2NL entries from CSV and return a dict mapping\n    (project_name, qualified_class_name, method_signature) to set of abstraction levels.\n    ';A:Dict[Tuple[str,str,str],Set[str]]={}
	with open(csv_path,'r',encoding=_C)as E:
		F=csv.DictReader(E)
		for B in F:
			C=B[_D],B['qualified_class_name'],B['method_signature'];D=B.get('abstraction_level','')
			if C not in A:A[C]=set()
			if D:A[C].add(D)
	return A
def get_all_tests_by_bucket(dataset:NL2TestDataset)->Dict[str,List[Test]]:'Return tests organized by bucket name.';A=dataset;return{_J:A.tests_with_one_focal_methods,_K:A.tests_with_two_focal_methods,_L:A.tests_with_more_than_two_to_five_focal_methods,_M:A.tests_with_more_than_five_to_ten_focal_methods,_N:A.tests_with_more_than_ten_focal_methods}
@ray.remote
def find_missing_tests_for_project(project_name:str,bucketed_file:str,test2nl_entries:Dict[Tuple[str,str,str],Set[str]],output_dir:str)->Dict[str,Any]|_B:
	"\n    Find tests in the bucketed dataset that are missing from Test2NL entries.\n\n    A test is considered missing if:\n    1. It doesn't exist in Test2NL at all, or\n    2. It doesn't have all three abstraction levels (low, medium, high)\n\n    Returns a dict with project stats or None if processing fails.\n    ";G=test2nl_entries;B=project_name
	with open(bucketed_file,'r',encoding=_C)as E:N=json.load(E)
	O=NL2TestDataset(**N);H=get_all_tests_by_bucket(O);A:Dict[str,List[Test]]={A:[]for A in H};F=0;C=0;I=0;J=0
	for(K,P)in H.items():
		for D in P:
			F+=1;L=B,D.qualified_class_name,D.method_signature
			if L not in G:A[K].append(D);C+=1;I+=1
			elif G[L]!=REQUIRED_ABSTRACTION_LEVELS:A[K].append(D);C+=1;J+=1
	if C==0:return{_D:B,_A:F,_H:0,_E:0,_F:0}
	Q=NL2TestDataset(dataset_name=f"{B}_missing",tests_with_one_focal_methods=A[_J],tests_with_two_focal_methods=A[_K],tests_with_more_than_two_to_five_focal_methods=A[_L],tests_with_more_than_five_to_ten_focal_methods=A[_M],tests_with_more_than_ten_focal_methods=A[_N]);M=Path(output_dir)/B;M.mkdir(parents=_G,exist_ok=_G);R=M/OUTPUT_FILE
	with open(R,'w',encoding=_C)as E:json.dump(Q.model_dump(),E,indent=2)
	return{_D:B,_A:F,_H:C,_E:I,_F:J}
def create_summary(output_dir:Path,results:List[Dict[str,Any]|_B])->_B:
	'Create a summary.json with missing test stats per project.';I='total_incomplete_abstractions';H='total_completely_missing';G='total_missing_tests';F='projects_with_missing';E='total_projects';C='projects';A:Dict[str,Any]={E:0,F:0,G:0,H:0,I:0,_A:0,C:{}}
	for B in results:
		if B is _B:continue
		A[E]+=1;A[_A]+=B[_A];D=B[_H];J=B[_E];K=B[_F]
		if D>0:A[F]+=1;A[G]+=D;A[H]+=J;A[I]+=K;A[C][B[_D]]={'total':B[_A],'missing':D,_E:J,_F:K}
	A[C]=dict(sorted(A[C].items()));L=output_dir/'summary.json'
	with open(L,'w',encoding=_C)as M:json.dump(A,M,indent=2)
	print('\nMissing tests analysis complete:');print(f"  Total projects: {A[E]}");print(f"  Projects with missing tests: {A[F]}");print(f"  Total tests: {A[_A]}");print(f"  Total missing tests: {A[G]}");print(f"    - Completely missing: {A[H]}");print(f"    - Incomplete abstractions: {A[I]}");print(f"  Summary saved to: {L}")
def process_projects(test2nl_file:Path,bucketed_dir:Path,output_dir:Path)->_B:
	'Process all projects in parallel to find missing tests.';F=bucketed_dir;E=test2nl_file;A=output_dir
	if A.exists():shutil.rmtree(A)
	A.mkdir(parents=_G,exist_ok=_G);print(f"Loading Test2NL entries from {E}...");D=load_test2nl_entries(E);K=sum(1 for A in D.values()if A==REQUIRED_ABSTRACTION_LEVELS);print(f"  Loaded {len(D)} unique test entries");print(f"  {K} entries have all abstraction levels");L=ray.put(D);B=[]
	for C in sorted(os.listdir(F)):
		if C.startswith('.')or C.startswith('__'):continue
		G=F/C
		if not G.is_dir():continue
		H=G/BUCKETED_FILE
		if not H.exists():print(f"Skipping {C}: no {BUCKETED_FILE} found");continue
		B.append(find_missing_tests_for_project.remote(C,str(H),L,str(A)))
	I:List[Dict[str,Any]|_B]=[]
	with tqdm(total=len(B),desc='Analyzing projects...')as M:
		while B:J,B=ray.wait(B,num_returns=1);N=ray.get(J);I.extend(N);M.update(len(J))
	create_summary(A,I)
def main():
	A=ROOT_DIR/TEST2NL_DIR/TEST2NL_FILE;B=ROOT_DIR/BUCKETED_DATASET_DIR;C=ROOT_DIR/MISSING_TESTS_DIR
	if not A.exists():raise FileNotFoundError(f"Test2NL file not found: {A}")
	if not B.exists():raise FileNotFoundError(f"Bucketed dataset directory not found: {B}")
	process_projects(A,B,C)
if __name__=='__main__':main()