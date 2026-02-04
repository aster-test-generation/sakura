_C='roundtrip_tests.json'
_B='descriptions.json'
_A=None
from pathlib import Path
from typing import List
from cldk.analysis.java import JavaAnalysis
from sakura.test2nl.evaluation import RoundTripEvaluator
from sakura.test2nl.generation import DescriptionGenerator,RoundTripGenerator
from sakura.test2nl.model.models import AbstractionLevel,RoundTripTest,Test2NLEntry,TestDescriptionInfo
from sakura.utils.analysis import CommonAnalysis
from sakura.utils.file_io import StructuredDataManager,TestFileInfo,TestFileManager
from sakura.utils.models import Method
class Pipeline:
	def __init__(A,analysis:JavaAnalysis,project_name:str,output_dir:Path,project_root:Path|_A=_A):
		D=project_root;C=output_dir;B=analysis;A.project_name=project_name;A.data_manager=StructuredDataManager(C);G,E,F=CommonAnalysis(B).categorize_classes();A.desc_generator=DescriptionGenerator(B,E,F);A.rt_generator=RoundTripGenerator(B)
		if D:A.rt_evaluator=RoundTripEvaluator(D,C)
		else:A.rt_evaluator=_A
	def _next_description_id(C)->int:
		'Ensure unique IDs across abstraction runs.'
		try:D=C.data_manager.load(_B,TestDescriptionInfo)
		except FileNotFoundError:return 1
		A=0
		for E in D:
			try:
				B=int(getattr(E,'id',0)or 0)
				if B>A:A=B
			except(TypeError,ValueError):continue
		return A+1
	def run_descriptions_of_project(B,abs_level:AbstractionLevel,num_trials:int=1,max_entries:int=0,only_interesting_tests:bool=False)->List[TestDescriptionInfo]:
		H='append';C=max_entries;A:List[TestDescriptionInfo]=[]
		for I in range(1,num_trials+1):
			J=C-len(A)if C>0 else 0
			if C>0 and len(A)>=C:break
			E=B.desc_generator.generate(abs_level,J,only_interesting_tests)
			for D in E:D.trial_number=I
			A.extend(E)
		F=B._next_description_id();G:List[Test2NLEntry]=[]
		for D in A:D.id=F;K=Test2NLEntry.from_test_description_info(D,B.project_name);G.append(K);F+=1
		B.data_manager.save(_B,A,format='json',mode=H);B.data_manager.save('test2nl.csv',G,format='csv',mode=H);return A
	def run_roundtrip(A)->List[RoundTripTest]:C=A.data_manager.load(_B,TestDescriptionInfo);B=A.rt_generator.generate(C);A.data_manager.save(_C,B);return B
	def run_rt_evaluation(A,gen_classes:bool=True):
		B=A.data_manager.load(_C,RoundTripTest)
		if gen_classes:C=[TestFileInfo.from_roundtrip_test(A)for A in B];TestFileManager(A.rt_evaluator.project_root).save_batch(C)
		A.rt_evaluator.reanalyze();D=A.rt_evaluator.grade(B);A.data_manager.save(_C,D)
	def run_descriptions_on_select(C,test_methods:List[Method],start_id:int=0)->tuple[List[Test2NLEntry],List[TestDescriptionInfo],int]:
		D:List[TestDescriptionInfo]=[];E:List[Test2NLEntry]=[];B=start_id
		for F in test_methods:
			for G in AbstractionLevel:
				A=C.desc_generator.generate_for_method(F.method_signature,F.qualified_class_name,G)
				if not A:continue
				A.id=B;D.append(A);H=Test2NLEntry.from_test_description_info(A,C.project_name);E.append(H);B+=1
		return E,D,B
	def run_description_of_method(B,select_method:Method,id:int,abstraction:AbstractionLevel)->tuple[Test2NLEntry|_A,TestDescriptionInfo|_A]:
		C=select_method;A=B.desc_generator.generate_for_method(C.method_signature,C.qualified_class_name,abstraction)
		if not A:return _A,_A
		A.id=id;D=Test2NLEntry.from_test_description_info(A,B.project_name);return D,A