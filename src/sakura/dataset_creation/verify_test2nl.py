_D='complete'
_C='total_methods'
_B='incomplete_levels'
_A='missing_completely'
import csv,json,os
from pathlib import Path
from typing import Dict,List,Set,Tuple
from sakura.dataset_creation.model import NL2TestDataset,Test
ROOT_DIR=Path(__file__).resolve().parent.parent.parent.parent
TEST2NL_DIR='resources/test2nl/filtered_dataset'
TEST2NL_FILE='test2nl.csv'
BUCKETED_TESTS_DIR='resources/filtered_bucketed_tests'
BUCKETED_FILE='nl2test.json'
REQUIRED_ABSTRACTION_LEVELS={'low','medium','high'}
def load_test2nl_ids_and_entries(csv_path:Path)->Tuple[List[int],Dict[Tuple[str,str,str],Set[str]]]:
	'\n    Load Test2NL entries from CSV and return:\n    1. List of all IDs in order\n    2. Dict mapping (project_name, qualified_class_name, method_signature)\n       to set of abstraction levels\n    ';D:List[int]=[];B:Dict[Tuple[str,str,str],Set[str]]={}
	with open(csv_path,'r',encoding='utf-8')as F:
		G=csv.DictReader(F)
		for A in G:
			D.append(int(A['id']));C=A['project_name'],A['qualified_class_name'],A['method_signature'];E=A.get('abstraction_level','')
			if C not in B:B[C]=set()
			if E:B[C].add(E)
	return D,B
def verify_contiguous_ids(ids:List[int])->Tuple[bool,List[str]]:
	'Verify IDs start at 0 and are contiguous with no gaps.';G='...';B:List[str]=[]
	if not ids:B.append('No IDs found in dataset');return False,B
	A=sorted(ids)
	if A[0]!=0:B.append(f"IDs do not start at 0 (first ID: {A[0]})")
	D=list(range(A[0],A[-1]+1))
	if A!=D:
		C=set(D)-set(A);E=[B for B in A if A.count(B)>1]
		if C:B.append(f"Missing IDs: {sorted(C)[:20]}{G if len(C)>20 else''}")
		if E:F=sorted(set(E));B.append(f"Duplicate IDs: {F[:20]}{G if len(F)>20 else''}")
	return len(B)==0,B
def get_all_tests(dataset:NL2TestDataset)->List[Test]:'Return all tests from all buckets.';A=dataset;return A.tests_with_one_focal_methods+A.tests_with_two_focal_methods+A.tests_with_more_than_two_to_five_focal_methods+A.tests_with_more_than_five_to_ten_focal_methods+A.tests_with_more_than_ten_focal_methods
def load_bucketed_methods(bucketed_dir:Path)->Set[Tuple[str,str,str]]:
	'Load all method keys from all bucketed test datasets.';B=bucketed_dir;C:Set[Tuple[str,str,str]]=set()
	for A in sorted(os.listdir(B)):
		if A.startswith('.')or A.startswith('__'):continue
		D=B/A
		if not D.is_dir():continue
		E=D/BUCKETED_FILE
		if not E.exists():continue
		with open(E,'r',encoding='utf-8')as G:H=json.load(G)
		I=NL2TestDataset(**H)
		for F in get_all_tests(I):J=A,F.qualified_class_name,F.method_signature;C.add(J)
	return C
def verify_abstraction_levels(test2nl_entries:Dict[Tuple[str,str,str],Set[str]],bucketed_methods:Set[Tuple[str,str,str]])->Tuple[bool,List[str],Dict[str,int]]:
	'Verify all bucketed methods have all three abstraction levels in Test2NL.';H=bucketed_methods;G=test2nl_entries;B:List[str]=[];A={_C:len(H),_D:0,_A:0,_B:0};D:List[Tuple[str,str,str]]=[];E:List[Tuple[Tuple[str,str,str],Set[str]]]=[]
	for F in H:
		if F not in G:A[_A]+=1;D.append(F)
		elif G[F]!=REQUIRED_ABSTRACTION_LEVELS:A[_B]+=1;E.append((F,G[F]))
		else:A[_D]+=1
	if D:
		B.append(f"Methods completely missing from Test2NL: {A[_A]}")
		for C in D[:5]:B.append(f"  - {C[0]}:{C[1]}#{C[2]}")
		if len(D)>5:B.append(f"  ... and {len(D)-5} more")
	if E:
		B.append(f"Methods with incomplete abstraction levels: {A[_B]}")
		for(C,I)in E[:5]:J=REQUIRED_ABSTRACTION_LEVELS-I;B.append(f"  - {C[0]}:{C[1]}#{C[2]} (missing: {J})")
		if len(E)>5:B.append(f"  ... and {len(E)-5} more")
	K=A[_A]==0 and A[_B]==0;return K,B,A
def main():
	L='   FAIL:';E='=';B=ROOT_DIR/TEST2NL_DIR/TEST2NL_FILE;C=ROOT_DIR/BUCKETED_TESTS_DIR
	if not B.exists():raise FileNotFoundError(f"Test2NL file not found: {B}")
	if not C.exists():raise FileNotFoundError(f"Bucketed tests directory not found: {C}")
	print(f"Loading Test2NL from {B}...");D,F=load_test2nl_ids_and_entries(B);print(f"  Loaded {len(D)} entries ({len(F)} unique methods)");print(f"\nLoading bucketed tests from {C}...");H=load_bucketed_methods(C);print(f"  Loaded {len(H)} methods");print('\n'+E*60);print('VERIFICATION RESULTS');print(E*60);print('\n1. Checking ID contiguity...');I,M=verify_contiguous_ids(D)
	if I:print(f"   PASS: IDs are contiguous (0 to {max(D)})")
	else:
		print(L)
		for G in M:print(f"   - {G}")
	print('\n2. Checking abstraction level completeness...');J,N,A=verify_abstraction_levels(F,H)
	if J:print(f"   PASS: All {A[_C]} methods have all abstraction levels")
	else:
		print(L)
		for G in N:print(f"   {G}")
	print('\n'+E*60);print('SUMMARY');print(E*60);print(f"  Total Test2NL entries: {len(D)}");print(f"  Unique methods in Test2NL: {len(F)}");print(f"  Methods in bucketed dataset: {A[_C]}");print(f"  Complete (all 3 levels): {A[_D]}");print(f"  Missing completely: {A[_A]}");print(f"  Incomplete levels: {A[_B]}");K=I and J;print(f"\nOverall: {'PASS'if K else'FAIL'}");return 0 if K else 1
if __name__=='__main__':exit(main())