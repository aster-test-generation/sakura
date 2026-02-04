_B='__pycache__'
_A=True
from pathlib import Path
import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from hamster.code_analysis.test_statistics import ProjectAnalysisInfo
from tqdm import tqdm
ROOT_DIR=Path(__file__).resolve().parent.parent.parent.parent
RESOURCES_DIR='resources/'
DATASETS_DIR='datasets/'
ANALYSIS_DIR='analysis/'
HAMSTER_DIR='hamster/'
OUTPUT_FILE_NAME='hamster.json'
def get_subfolders(base_folder:Path)->list[str]:'Get all immediate subfolders in the base folder.';B=base_folder;return[str(B/A.name)for A in B.iterdir()if A.is_dir()and A.name!=_B and not A.name.startswith('.')]
@ray.remote
def _create_hamster_model(project_path:str,analysis_path:str,output_path:str)->None:
	'Create a hamster model from CLDK analysis for a project.';A=project_path;B=Path(A).name
	if B==_B or B.startswith('.'):return
	try:
		D=CLDK(language='java').analysis(project_path=A,analysis_backend_path=None,analysis_level=AnalysisLevel.symbol_table,analysis_json_path=analysis_path,eager=_A);E=ProjectAnalysisInfo(analysis=D,dataset_name=B).gather_project_analysis_info();F=E.model_dump_json();C=Path(output_path);C.mkdir(parents=_A,exist_ok=_A)
		with open(C/OUTPUT_FILE_NAME,'w')as G:G.write(F)
	except Exception as H:print(f"Error processing dataset {A}: {H}")
def main()->None:
	'Process all projects and create hamster models.';B=ROOT_DIR/RESOURCES_DIR;C=B/DATASETS_DIR;D=B/ANALYSIS_DIR;E=B/HAMSTER_DIR;C.mkdir(parents=_A,exist_ok=_A);D.mkdir(parents=_A,exist_ok=_A);E.mkdir(parents=_A,exist_ok=_A);I=get_subfolders(C);A=[]
	for F in I:G=Path(F).name;A.append(_create_hamster_model.remote(F,str(D/G),str(E/G)))
	J=[]
	with tqdm(total=len(A),desc='Processing folders')as K:
		while A:H,A=ray.wait(A,num_returns=1);L=ray.get(H);J.extend(L);K.update(len(H))
if __name__=='__main__':main()