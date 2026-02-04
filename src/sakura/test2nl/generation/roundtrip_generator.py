from typing import List
from cldk.analysis.java import JavaAnalysis
from sakura.test2nl.model.models import TestDescriptionInfo,RoundTripTest
from sakura.test2nl.prompts import RoundTripPrompt
from sakura.utils.config import Config
class RoundTripGenerator:
	def __init__(A,analysis:JavaAnalysis)->None:B=Config();A.temp=B.get('llm','code_gen_temp');A.roundtrip_prompt=RoundTripPrompt(analysis)
	def generate(B,test_descriptions:List[TestDescriptionInfo])->List[RoundTripTest]:
		C:List[RoundTripTest]=[]
		for A in test_descriptions:
			D,E,F=B.roundtrip_prompt.generate(A.method_signature,A.qualified_class_name,A.description)
			if F:C.append(RoundTripTest(prompt=E,temperature=B.temp,generated_test=D,method_signature=A.method_signature,qualified_class_name=A.qualified_class_name,generated_description=A))
		return C