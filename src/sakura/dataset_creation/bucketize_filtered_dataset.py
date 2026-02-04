_A='nl2test.json'
import json,os
from pathlib import Path
from typing import Dict,List
from sakura.dataset_creation.model import NL2TestDataset,Test
ROOT_DIR=Path(__file__).resolve().parent.parent.parent.parent
INPUT_FILE_NAME=_A
OUTPUT_FILE_NAME=_A
SUMMARY_FILE_NAME='summary.json'
RESOURCES_DIR='resources'
BUCKETED_TESTS_DIR='bucketed_tests'
FILTERED_TESTS_DIR='filtered_tests'
FILTERED_BUCKETED_TESTS_DIR='filtered_bucketed_tests'
def filter_tests_by_date(dataset:NL2TestDataset,test_classes_and_methods:Dict[str,List[str]])->NL2TestDataset:
	'\n    Filter tests from an NL2TestDataset to only include those present in filtered tests.\n\n    Args:\n        dataset: The original NL2TestDataset to filter.\n        test_classes_and_methods: Dict mapping qualified class names to method signatures.\n\n    Returns:\n        New NL2TestDataset containing only tests present in test_classes_and_methods.\n    ';C=test_classes_and_methods;A=dataset
	def B(tests:List[Test])->List[Test]:
		B=[]
		for A in tests:
			if A.qualified_class_name in C:
				D=C[A.qualified_class_name]
				if A.method_signature in D:B.append(A)
		return B
	return NL2TestDataset(dataset_name=A.dataset_name,tests_with_one_focal_methods=B(A.tests_with_one_focal_methods),tests_with_two_focal_methods=B(A.tests_with_two_focal_methods),tests_with_more_than_two_to_five_focal_methods=B(A.tests_with_more_than_two_to_five_focal_methods),tests_with_more_than_five_to_ten_focal_methods=B(A.tests_with_more_than_five_to_ten_focal_methods),tests_with_more_than_ten_focal_methods=B(A.tests_with_more_than_ten_focal_methods))
def process_projects(bucket_dir:Path,filtered_dir:Path,output_dir:Path)->None:
	'\n    Process all projects in bucket_dir and filter them against filtered_dir.\n\n    Args:\n        bucket_dir: Directory containing bucketed NL2TestDataset files per project.\n        filtered_dir: Directory containing filtered test classes and methods per project.\n        output_dir: Directory to write filtered bucketed datasets.\n    ';Y='tests_with_more_than_ten_focal_methods';X='tests_with_more_than_five_to_ten_focal_methods';W='tests_with_more_than_two_to_five_focal_methods';V='tests_with_two_focal_methods';U='tests_with_one_focal_methods';J=bucket_dir;G=output_dir;F='utf-8';E='total';D=True;G.mkdir(parents=D,exist_ok=D);K=[U,V,W,X,Y];C:Dict[str,int]={A:0 for A in K};C[E]=0;H:Dict[str,Dict[str,int]]={}
	def Z(dataset:NL2TestDataset)->Dict[str,int]:'Get per-bucket test counts for a dataset.';A=dataset;B={U:len(A.tests_with_one_focal_methods),V:len(A.tests_with_two_focal_methods),W:len(A.tests_with_more_than_two_to_five_focal_methods),X:len(A.tests_with_more_than_five_to_ten_focal_methods),Y:len(A.tests_with_more_than_ten_focal_methods)};B[E]=sum(B.values());return B
	for A in os.listdir(J):
		if A.startswith('.')or A.startswith('__'):continue
		L=J/A
		if not L.is_dir():continue
		M=filtered_dir/A
		if not M.exists():print(f"Skipping {A}: not found in filtered_tests/");continue
		N=L/INPUT_FILE_NAME
		if not N.exists():print(f"Skipping {A}: no {INPUT_FILE_NAME} in bucketed_tests/");continue
		with open(N,'r',encoding=F)as B:a=json.load(B)
		b=NL2TestDataset(**a);O=M/INPUT_FILE_NAME
		if not O.exists():print(f"Skipping {A}: no {INPUT_FILE_NAME} in filtered_tests/");continue
		with open(O,'r',encoding=F)as B:c=json.load(B)
		d=c.get('new_tests',{});P=filter_tests_by_date(b,d);Q=G/A;Q.mkdir(parents=D,exist_ok=D);R=Q/OUTPUT_FILE_NAME
		with open(R,'w',encoding=F)as B:json.dump(P.model_dump(),B,indent=4)
		I=Z(P);H[A]=I
		for S in K:C[S]+=I[S]
		C[E]+=I[E];print(f"Processed {A}: saved to {R}")
	T=G/SUMMARY_FILE_NAME;e={'totals':C,'projects':dict(sorted(H.items()))}
	with open(T,'w',encoding=F)as B:json.dump(e,B,indent=2)
	print(f"\nTop-level summary saved to: {T}");print(f"  - Total projects: {len(H)}")
def main():A=ROOT_DIR/RESOURCES_DIR;B=A/BUCKETED_TESTS_DIR;C=A/FILTERED_TESTS_DIR;D=A/FILTERED_BUCKETED_TESTS_DIR;process_projects(B,C,D)
if __name__=='__main__':main()