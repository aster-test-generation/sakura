_A='nl2test.json'
import json,os
from pathlib import Path
import random
from sakura.utils import constants
ROOT_DIR=Path(__file__).resolve().parent.parent.parent.parent
INPUT_FILE_NAME=_A
OUTPUT_FILE_NAME=_A
RESOURCES_DIR='resources'
BUCKETED_TESTS_DIR='bucketed_tests'
SAMPLED_TESTS_DIR='sampled_tests'
RANDOM_SAMPLE_SIZE=20
class RandomSampleDataset:
	def __init__(A):0
	@staticmethod
	def create_random_sample_dataset(dataset_folder:str):
		D=dataset_folder
		with open(Path(D).joinpath(INPUT_FILE_NAME),'r')as B:G=json.load(B)
		C={}
		for(E,A)in G.items():
			if isinstance(A,list):C[E]=random.sample(A,min(RANDOM_SAMPLE_SIZE,len(A)))
			else:C[E]=A
		F=ROOT_DIR/RESOURCES_DIR/SAMPLED_TESTS_DIR/Path(D).name;os.makedirs(F,exist_ok=True)
		with open(F/OUTPUT_FILE_NAME,'w')as B:json.dump(C,B,indent=2)
	@staticmethod
	def get_subfolders(base_folder):'Get all immediate subfolders in the base folder.';A=base_folder;return[os.path.join(A,B)for B in os.listdir(A)if os.path.isdir(os.path.join(A,B))]
if __name__=='__main__':
	projects=RandomSampleDataset.get_subfolders(ROOT_DIR/RESOURCES_DIR/BUCKETED_TESTS_DIR)
	for project in projects:RandomSampleDataset.create_random_sample_dataset(project)