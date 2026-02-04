_F='is_bdd'
_E='method_signature'
_D='qualified_class_name'
_C='project_name'
_B='description'
_A='abstraction_level'
import csv
from pathlib import Path
from typing import List
from sakura.test2nl.model.models import AbstractionLevel,Test2NLEntry
ROOT_DIR=Path(__file__).resolve().parent.parent.parent.parent
FIRST_DATASET_DIR='resources/test2nl/old_filtered_dataset'
SECOND_DATASET_DIR='resources/test2nl/missing_dataset'
NEW_DATASET_DIR='resources/test2nl/filtered_dataset'
TEST2NL_FILE='test2nl.csv'
def load_test2nl_entries(csv_path:Path)->List[Test2NLEntry]:
	'Load Test2NL entries from a CSV file.';B:List[Test2NLEntry]=[]
	with open(csv_path,'r',encoding='utf-8')as D:
		E=csv.DictReader(D)
		for A in E:
			C=None
			if A.get(_A):C=AbstractionLevel(A[_A])
			F=Test2NLEntry(id=int(A['id']),description=A[_B],project_name=A[_C],qualified_class_name=A[_D],method_signature=A[_E],abstraction_level=C,is_bdd=A.get(_F,'False').lower()=='true');B.append(F)
	return B
def save_test2nl_entries(entries:List[Test2NLEntry],csv_path:Path)->None:
	'Save Test2NL entries to a CSV file.';B=csv_path;B.parent.mkdir(parents=True,exist_ok=True);D=[_A,_B,'id',_F,_E,_C,_D]
	with open(B,'w',encoding='utf-8',newline='')as E:
		C=csv.DictWriter(E,fieldnames=D);C.writeheader()
		for A in entries:C.writerow({_A:A.abstraction_level.value if A.abstraction_level else'',_B:A.description,'id':A.id,_F:A.is_bdd,_E:A.method_signature,_C:A.project_name,_D:A.qualified_class_name})
def get_entry_key(entry:Test2NLEntry)->tuple:'Return the unique key for an entry (excluding id and is_bdd).';A=entry;B=A.abstraction_level.value if A.abstraction_level else None;return A.project_name,A.qualified_class_name,A.method_signature,B
def main():
	D=ROOT_DIR/FIRST_DATASET_DIR/TEST2NL_FILE;E=ROOT_DIR/SECOND_DATASET_DIR/TEST2NL_FILE;H=ROOT_DIR/NEW_DATASET_DIR/TEST2NL_FILE
	if not D.exists():raise FileNotFoundError(f"First dataset not found: {D}")
	if not E.exists():raise FileNotFoundError(f"Second dataset not found: {E}")
	print(f"Loading first dataset from {D}...");B=load_test2nl_entries(D);print(f"  Loaded {len(B)} entries");print(f"Loading second dataset from {E}...");L=load_test2nl_entries(E);print(f"  Loaded {len(L)} entries");I:dict[tuple,int]={}
	for(F,A)in enumerate(B):I[get_entry_key(A)]=F
	J=0;C:List[Test2NLEntry]=[]
	for A in L:
		M=get_entry_key(A)
		if M in I:F=I[M];O=B[F].id;A.id=O;B[F]=A;J+=1
		else:C.append(A)
	print(f"  Replaced {J} entries in first dataset");print(f"  Found {len(C)} new unique entries to add");N=max(A.id for A in B);print(f"  Max ID in first dataset: {N}")
	for(P,A)in enumerate(C):A.id=N+1+P
	G=B+C;print(f"  Merged dataset contains {len(G)} entries");K=sorted(A.id for A in G);Q=list(range(K[0],K[-1]+1))
	if K!=Q:print('  Warning: IDs are not contiguous')
	print(f"Saving merged dataset to {H}...");save_test2nl_entries(G,H);print('  Done!');print('\nMerge complete:');print(f"  First dataset: {len(B)} entries");print(f"  Entries replaced: {J}");print(f"  New entries added: {len(C)}");print(f"  Merged dataset: {len(G)} entries");print(f"  Output: {H}")
if __name__=='__main__':main()