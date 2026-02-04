_E='new_test_method_count'
_D='new_test_class_count'
_C=False
_B='summary'
_A=True
import json,os,subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict,List,Optional,Set,Tuple
import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from cldk.analysis.java import JavaAnalysis
from tqdm import tqdm
from sakura.utils.analysis.java_analyzer import CommonAnalysis
ROOT_DIR=Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_FILE_NAME='nl2test.json'
SUMMARY_FILE_NAME='summary.json'
DEFAULT_DATE_STR='2025-01-31'
RESOURCES_DIR='resources'
DATASETS_DIR='datasets'
FILTERED_TESTS_DIR='filtered_tests'
ANALYSIS_DIR='analysis'
class FilterByDate:
	def _date_str_to_timestamp(B,date_str:str)->int:'Convert YYYY-MM-DD to Unix timestamp.';A=datetime.strptime(date_str,'%Y-%m-%d');return int(A.timestamp())
	def _get_line_timestamps(G,repo_path:str,file_path:str)->Dict[int,int]:
		'Get timestamp for each line using git blame.';F=['git','-C',repo_path,'blame','--line-porcelain',file_path];B=subprocess.run(F,capture_output=_A,text=_A)
		if B.returncode!=0:return{}
		C:Dict[int,int]={};D=0;E=0
		for A in B.stdout.splitlines():
			if A.startswith('author-time '):E=int(A.split()[1])
			elif A.startswith('\t'):D+=1;C[D]=E
		return C
	def get_new_methods_via_blame(B,repo_path:str,date_str:str,test_files:List[str],analysis:JavaAnalysis,test_class_methods:Dict[str,List[str]])->Dict[str,List[str]]:
		'\n        Find test methods added after a date by checking git blame timestamps.\n\n        Args:\n            repo_path: Path to the repository\n            date_str: Cutoff date in YYYY-MM-DD format\n            test_files: List of absolute paths to test files\n            analysis: CLDK JavaAnalysis instance\n            test_class_methods: Mapping of test class names to their test method signatures\n\n        Returns:\n            Mapping of qualified class names to lists of new test method signatures\n        ';D=repo_path;H=B._date_str_to_timestamp(date_str);A:Dict[str,List[str]]={}
		for E in test_files:
			I=os.path.relpath(E,D);F=B._get_line_timestamps(D,I)
			if not F:continue
			J=B._find_new_methods(E,F,H,analysis,test_class_methods)
			for(C,G)in J.items():
				if C in A:A[C].extend(G)
				else:A[C]=G
		return A
	def _find_new_methods(N,file_path:str,line_timestamps:Dict[int,int],cutoff_timestamp:int,analysis:JavaAnalysis,test_class_methods:Dict[str,List[str]])->Dict[str,List[str]]:
		'\n        Find test methods where all lines were added after the cutoff date.\n\n        Args:\n            file_path: Absolute path to the Java file\n            line_timestamps: Mapping of line numbers (1-indexed) to Unix timestamps\n            cutoff_timestamp: Unix timestamp for the cutoff date\n            analysis: CLDK JavaAnalysis instance\n            test_class_methods: Mapping of test class names to their test method signatures\n\n        Returns:\n            Mapping of qualified class names to lists of new test method signatures\n        ';F=test_class_methods;C=analysis;B:Dict[str,List[str]]={};G=C.get_java_compilation_unit(file_path)
		if not G:return B
		for A in G.type_declarations.keys():
			if A not in F:continue
			I=set(F[A])
			for D in C.get_methods_in_class(A):
				if D not in I:continue
				E=C.get_method(A,D)
				if not E:continue
				J=E.start_line;K=E.end_line;H=_A
				for L in range(J,K+1):
					M=line_timestamps.get(L,0)
					if M<cutoff_timestamp:H=_C;break
				if H:
					if A not in B:B[A]=[]
					B[A].append(D)
		return B
	def process_repos_in_dir(P,base_dir:str,date_str:str,analysis_dir:Path,debug:bool=_C)->Dict:
		'Iterate through subdirectories and collect data.';E=base_dir
		if not ray.is_initialized():
			if debug:print('Running Ray in local debug mode...');ray.init(local_mode=_A,ignore_reinit_error=_A)
			else:ray.init(ignore_reinit_error=_A)
		A=[]
		for B in os.listdir(E):
			C=os.path.join(E,B)
			if not os.path.isdir(C)or not os.path.exists(os.path.join(C,'.git')):continue
			M=_process_single_repo.remote(B,C,date_str,str(analysis_dir));A.append(M)
		F=[]
		with tqdm(total=len(A),desc='Processing repositories')as N:
			while A:G,A=ray.wait(A,num_returns=1);O=ray.get(G);F.extend(O);N.update(len(G))
		D:Dict={};H=0;I=0
		for J in F:
			if J is None:continue
			B,K=J;D[B]=K;L=K.get(_B,{});H+=L.get(_D,0);I+=L.get(_E,0)
		D[_B]={'total_new_test_classes':H,'total_new_test_methods':I};return D
	def save_results(J,results:Dict,output_dir:Path)->None:
		'Save per-project and summary results.';H='utf-8';C=results;A=output_dir;A.mkdir(parents=_A,exist_ok=_A);I=C.pop(_B,{});D={}
		for(E,F)in C.items():
			G=A/E;G.mkdir(parents=_A,exist_ok=_A)
			with open(G/OUTPUT_FILE_NAME,'w',encoding=H)as B:json.dump(F,B,indent=4)
			D[E]=F.get(_B,{})
		with open(A/SUMMARY_FILE_NAME,'w',encoding=H)as B:json.dump({_B:I,'projects':D},B,indent=4)
		print(f"\nResults saved to: {A}")
RepoResult=Optional[Tuple[str,Dict]]
@ray.remote
def _process_single_repo(project_name:str,repo_path:str,date_str:str,analysis_dir:str)->RepoResult:
	'Process one repo to find tests added after a date.';D=repo_path;A=project_name;H=FilterByDate()
	try:
		I=Path(analysis_dir)/A;B=CLDK(language='java').analysis(project_path=D,analysis_backend_path=None,analysis_level=AnalysisLevel.symbol_table,analysis_json_path=I,eager=_C);J=CommonAnalysis(B);E,K,K=J.categorize_classes();F:Set[str]=set()
		for L in E.keys():
			G=B.get_java_file(qualified_class_name=L)
			if G:F.add(G)
		C=H.get_new_methods_via_blame(D,date_str,list(F),B,E);M={_B:{_D:len(C),_E:sum(len(A)for A in C.values())},'new_tests':C};return A,M
	except Exception as N:return A,{'error':str(N)}
def main():A=ROOT_DIR/RESOURCES_DIR;C=A/DATASETS_DIR;D=A/FILTERED_TESTS_DIR;E=DEFAULT_DATE_STR;F=_C;G=A/ANALYSIS_DIR;B=FilterByDate();H=B.process_repos_in_dir(str(C),E,analysis_dir=G,debug=F);B.save_results(H,D)
if __name__=='__main__':main()