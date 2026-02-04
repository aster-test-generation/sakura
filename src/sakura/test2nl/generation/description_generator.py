_A=None
from typing import Dict,List
from cldk.analysis.java import JavaAnalysis
from hamster.code_analysis.model.models import TestingFramework
from sakura.test2nl.model.models import AbstractionLevel,TestDescriptionInfo
from sakura.test2nl.prompts import Test2NLPrompt
from sakura.utils.analysis import CommonAnalysis,Reachability
class DescriptionGenerator:
	def __init__(A,analysis:JavaAnalysis,application_classes:List[str]|_A=_A,test_utility_classes:List[str]|_A=_A)->_A:C=test_utility_classes;B=application_classes;A.analysis=analysis;A._application_classes=B if B else[];A._test_utility_classes=C if C else[];A.test2nl_prompt=Test2NLPrompt(A.analysis,A._application_classes,A._test_utility_classes)
	def generate_for_method(F,method_signature:str,qualified_class_name:str,abstraction:AbstractionLevel=AbstractionLevel.HIGH)->TestDescriptionInfo|_A:
		C=qualified_class_name;B=method_signature;A=abstraction;D=A.get_temperature();E,G,H=F.test2nl_prompt.generate(B,C,A,temperature=D)
		if not H or E is _A:return
		I=TestDescriptionInfo(description=E,prompt=G,abstraction_level=A,temperature=D,method_signature=B,qualified_class_name=C);return I
	def generate_for_class(C,qualified_class_name:str,testing_frameworks:List[TestingFramework],abstraction:AbstractionLevel=AbstractionLevel.HIGH,max_entries:int=0)->List[TestDescriptionInfo]:
		E=max_entries;D=qualified_class_name;A:List[TestDescriptionInfo]=[];F:Dict[str,List[str]]=Reachability(C.analysis).get_reachable_test_methods(D,testing_frameworks)
		for(G,H)in F.items():
			for I in H:
				if E>0 and len(A)>=E:return A
				B=C.generate_for_method(I,G,abstraction)
				if B:B.qualified_class_name=D;A.append(B)
		return A
	def generate_for_project(E,abstraction:AbstractionLevel=AbstractionLevel.HIGH,max_entries:int=0,only_interesting_tests:bool=False)->List[TestDescriptionInfo]:
		G=abstraction;A=max_entries;C:List[TestDescriptionInfo]=[];D=CommonAnalysis(E.analysis)
		if only_interesting_tests:
			I=D.get_complicated_focal_tests()
			for(B,J)in I.items():
				if A>0 and len(C)>=A:break
				if D.is_abstract_class(B):continue
				F=D.get_testing_frameworks_for_class(B)
				for K in J:
					if A>0 and len(C)>=A:break
					H=E.generate_for_method(K,B,G)
					if H:C.append(H)
		else:
			for B in E.analysis.get_classes():
				if A>0 and len(C)>=A:break
				if D.is_abstract_class(B):continue
				F=D.get_testing_frameworks_for_class(B)
				if D.is_test_class(B,F):L=A-len(C)if A>0 else 0;M=E.generate_for_class(B,F,G,L);C.extend(M)
		return C
	def generate(A,abstraction_level:AbstractionLevel=AbstractionLevel.HIGH,max_entries:int=0,only_interesting_tests:bool=False)->List[TestDescriptionInfo]:return A.generate_for_project(abstraction_level,max_entries,only_interesting_tests)