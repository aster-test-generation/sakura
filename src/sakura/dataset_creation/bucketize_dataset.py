_I='tests_with_more_than_ten_focal_methods'
_H='tests_with_more_than_five_to_ten_focal_methods'
_G='tests_with_more_than_two_to_five_focal_methods'
_F='tests_with_two_focal_methods'
_E='tests_with_one_focal_methods'
_D='dataset_name'
_C=None
_B=False
_A=True
import json
from pathlib import Path
from typing import Any,Dict,List
import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from cldk.analysis.java import JavaAnalysis
from hamster.code_analysis.model.models import ProjectAnalysis
from tqdm import tqdm
from sakura.dataset_creation.create_hamster_model import get_subfolders
from sakura.dataset_creation.model import NL2TestDataset,Test
ROOT_DIR=Path(__file__).resolve().parent.parent.parent.parent
RESOURCES_DIR='resources'
DATASETS_DIR='datasets/'
HAMSTER_DIR='hamster/'
BUCKETED_DIR='bucketed_tests/'
ANALYSIS_DIR='analysis/'
INPUT_FILE_NAME='hamster.json'
OUTPUT_FILE_NAME='nl2test.json'
SUMMARY_FILE_NAME='summary.json'
def _should_skip_test_class(analysis:JavaAnalysis,qualified_class_name:str)->bool:
	'Determine if a test class should be skipped from bucketing.';A=analysis.get_class(qualified_class_name)
	if not A:return _A
	if A.modifiers and'abstract'in A.modifiers:return _A
	if A.is_interface:return _A
	if A.is_annotation_declaration:return _A
	if A.is_enum_declaration:return _A
	return _B
def _has_empty_body(code:str)->bool:
	'Check if method body contains only whitespace, braces, and/or comments.';E=code;D='*/';F=E.find('{');G=E.rfind('}')
	if F==-1 or G==-1 or F>=G:return _A
	J=E[F+1:G];H=_B
	for K in J.split('\n'):
		B=K.strip()
		if H:
			if D in B:
				H=_B;I=B[B.index(D)+2:].strip()
				if I and not I.startswith('//'):return _B
			continue
		if not B:continue
		A=B
		while'/*'in A:
			C=A.index('/*')
			if D in A[C:]:L=A.index(D,C)+2;A=(A[:C]+A[L:]).strip()
			else:H=_A;A=A[:C].strip();break
		if not A:continue
		if A.startswith('//')or A.startswith('*'):continue
		return _B
	return _A
def _should_skip_test_method(analysis:JavaAnalysis,qualified_class_name:str,method_signature:str)->bool:
	'Determine if a test method should be skipped from bucketing.';A=analysis.get_method(qualified_class_name,method_signature)
	if not A:return _A
	for C in A.annotations:
		B=C.lstrip('@').split('(')[0]
		if B.startswith('Disabled'):return _A
		if B=='TestFactory':return _A
	if _has_empty_body(A.code.strip()):return _A
	if not A.code.isascii():return _A
	return _B
@ray.remote
def _create_bucketized_dataset(hamster_path:str,output_path:str,analysis_path:str,project_path:str)->Dict[str,Any]|_C:
	'Create a bucketized dataset from hamster analysis, filtering invalid test classes.';Q='__pycache__';M=hamster_path;R=Path(M).parent;N=R.name
	if N==Q or N.startswith('.'):return
	with open(M,'r')as F:S=json.load(F);G=ProjectAnalysis.model_validate(S)
	if not G:return
	A=G.dataset_name
	if not A or A==Q or A.startswith('.'):return
	try:O=CLDK(language='java').analysis(project_path=project_path,analysis_backend_path=_C,analysis_level=AnalysisLevel.symbol_table,analysis_json_path=analysis_path,eager=_B)
	except Exception:return
	H:List[Test]=[];I:List[Test]=[];J:List[Test]=[];K:List[Test]=[];L:List[Test]=[]
	for D in G.test_class_analyses:
		if _should_skip_test_class(O,D.qualified_class_name):continue
		for E in D.test_method_analyses:
			if _should_skip_test_method(O,D.qualified_class_name,E.method_signature):continue
			T=E.focal_classes or[];B=sum(len(A.focal_method_names or[])for A in T);C=Test(qualified_class_name=D.qualified_class_name,method_signature=E.method_signature,focal_details=E.focal_classes)
			if B==1:H.append(C)
			elif B==2:I.append(C)
			elif 2<B<=5:J.append(C)
			elif 5<B<=10:K.append(C)
			elif B>10:L.append(C)
	U=NL2TestDataset(dataset_name=A,tests_with_one_focal_methods=H,tests_with_two_focal_methods=I,tests_with_more_than_two_to_five_focal_methods=J,tests_with_more_than_five_to_ten_focal_methods=K,tests_with_more_than_ten_focal_methods=L);P=Path(output_path);P.mkdir(parents=_A,exist_ok=_A)
	with open(P/OUTPUT_FILE_NAME,'w')as F:F.write(U.model_dump_json())
	return{_D:A,_E:len(H),_F:len(I),_G:len(J),_H:len(K),_I:len(L)}
def _create_summary(bucketed_dir:Path,results:List[Dict[str,Any]|_C])->_C:
	'Create a summary.json with test counts per project per bucket.';G='total';H=[_E,_F,_G,_H,_I];A={A:0 for A in H};A[G]=0;I:Dict[str,Dict[str,int]]={}
	for B in results:
		if B is _C:continue
		J=B[_D];C:Dict[str,int]={};D=0
		for E in H:F=B[E];C[E]=F;A[E]+=F;D+=F
		C[G]=D;A[G]+=D;I[J]=C
	K:Dict[str,Any]={'totals':A,'projects':dict(sorted(I.items()))}
	with open(bucketed_dir/SUMMARY_FILE_NAME,'w')as L:json.dump(K,L,indent=2)
def main():
	'Process all projects and create bucketized datasets.';B=ROOT_DIR/RESOURCES_DIR;J=B/DATASETS_DIR;E=B/HAMSTER_DIR;C=B/BUCKETED_DIR;F=B/ANALYSIS_DIR;E.mkdir(parents=_A,exist_ok=_A);C.mkdir(parents=_A,exist_ok=_A);F.mkdir(parents=_A,exist_ok=_A);K=get_subfolders(E);A=[]
	for G in K:D=Path(G).name;A.append(_create_bucketized_dataset.remote(str(Path(G)/INPUT_FILE_NAME),str(C/D),str(F/D),str(J/D)))
	H:List[Dict[str,Any]|_C]=[]
	with tqdm(total=len(A),desc='Processing projects...')as L:
		while A:I,A=ray.wait(A,num_returns=1);M=ray.get(I);H.extend(M);L.update(len(I))
	_create_summary(C,H)
if __name__=='__main__':main()