from typing import List,Tuple
import yaml
from cldk.analysis.java import JavaAnalysis
from sakura.test2nl.extractors import ClassExtractor
from sakura.test2nl.model.models import ClassContext
from sakura.test2nl.prompts.load_prompt import LoadPrompt,PromptFormat
from sakura.utils.analysis import CommonAnalysis
from sakura.utils.llm import ClientType,LLMClient
class RoundTripPrompt:
	'\n    This class is used to generate the roundtrip prompt for the test case.\n\n    TODO: Update prompt.\n    '
	def __init__(A,analysis:JavaAnalysis):super().__init__();A.analysis=analysis;A.llm=LLMClient(ClientType.CODE_GEN)
	def format(A,test_method_signature:str,test_qualified_class:str,description:str)->str:
		B=test_qualified_class;E=A.analysis.get_method(B,test_method_signature);C:List[ClassContext]=[];F:List[str]=CommonAnalysis(A.analysis).get_referenced_app_classes(E)
		for D in F:
			if D!=B:C.append(ClassExtractor(A.analysis).extract(D,complete_methods=True))
		G:List[str]=[yaml.dump(A.model_dump(),sort_keys=False,indent=4)for A in C];H=LoadPrompt.load_prompt('roundtrip_prompt.jinja2',PromptFormat.JINJA2);I=H.format(test_case_description=description,custom_classes=G);return I
	def generate(B,test_method_signature:str,test_qualified_class:str,description:str)->Tuple[str|None,str,bool]:
		A=B.format(test_method_signature,test_qualified_class,description);D=B.llm.generate(A,sanitize=True);C=D.strip()
		if C:return C,A,True
		return None,A,False