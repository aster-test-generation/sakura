from __future__ import annotations
_B='forbid'
_A=None
import textwrap
from enum import Enum
from typing import Annotated,List,Literal,Optional,Set,Tuple,Union
from pydantic import BaseModel,ConfigDict,Field
from sakura.utils.models import NL2TestInput
class DecompositionMode(Enum):GHERKIN='gherkin';GRAMMATICAL='grammatical'
class Step(BaseModel):id:int;task:str;uses:str;produces:str
class GherkinStep(BaseModel):given:List[Step];when:List[Step];then:List[Step]
_GHERKIN_STEPS_DESC=textwrap.dedent('\n    Ordered Gherkin-style step groups (given, when, then) that capture distinct behaviors or paths.\n    ').strip()
class Scenario(BaseModel):setup:List[Step];gherkin_groups:Annotated[List[GherkinStep],Field(description=_GHERKIN_STEPS_DESC)];teardown:List[Step]
class CandidateMethod(BaseModel):declaring_class_name:str;containing_class_name:str;method_signature:str
def _is_valid_candidate(candidate:CandidateMethod|_A)->bool:
	A=candidate
	if A is _A:return False
	return any(getattr(A,B,'').strip()for B in('declaring_class_name','containing_class_name','method_signature'))
def _candidate_key(candidate:CandidateMethod)->Tuple[str,str,str]:A=candidate;return(A.declaring_class_name or'').strip(),(A.containing_class_name or'').strip(),(A.method_signature or'').strip()
def _normalize_candidates(candidates:List[CandidateMethod],best_candidate:CandidateMethod|_A=_A,limit:int=3)->Tuple[List[CandidateMethod],CandidateMethod|_A]:
	C=limit;A=best_candidate
	if C<=0:return[],A
	D:List[CandidateMethod]=[];E:Set[Tuple[str,str,str]]=set()
	def F(candidate:CandidateMethod|_A)->_A:
		A=candidate
		if not _is_valid_candidate(A):return
		assert A is not _A;B=_candidate_key(A)
		if B in E:return
		E.add(B);D.append(A)
	F(A)
	for G in candidates:F(G)
	B=D[:C]
	if B:return B,B[0]
	return[],A
class ArgBinding(BaseModel):arg_name:str;arg_value:str
class LocalizedStep(Step):
	candidate_methods:List[CandidateMethod];arg_bindings:List[ArgBinding];comments:str;external:bool
	def enforce_candidate_limit(A,limit:int=3)->_A:B,C=_normalize_candidates(A.candidate_methods,limit=limit);A.candidate_methods=B
class LocalizedGherkinStep(BaseModel):given:List[LocalizedStep];when:List[LocalizedStep];then:List[LocalizedStep]
class LocalizedScenario(BaseModel):
	setup:List[LocalizedStep];gherkin_groups:Annotated[List[LocalizedGherkinStep],Field(description=_GHERKIN_STEPS_DESC)];teardown:List[LocalizedStep]
	@classmethod
	def from_scenario(E,scenario:Scenario)->'LocalizedScenario':
		'Create a LocalizedScenario from a plain Scenario.\n\n        Fields not present in Scenario are initialized with empty defaults so the\n        localization agent can fill them in later.\n        ';B=scenario
		def A(s:Step)->LocalizedStep:return LocalizedStep(id=s.id,task=s.task,uses=s.uses,produces=s.produces,candidate_methods=[],arg_bindings=[],comments='',external=False)
		D:List[LocalizedGherkinStep]=[]
		for C in B.gherkin_groups:D.append(LocalizedGherkinStep(given=[A(B)for B in C.given],when=[A(B)for B in C.when],then=[A(B)for B in C.then]))
		return E(setup=[A(B)for B in B.setup],gherkin_groups=D,teardown=[A(B)for B in B.teardown])
	def enforce_candidate_limits(B,limit:int=3)->_A:
		C=limit
		for A in B.setup:A.enforce_candidate_limit(C)
		for D in B.gherkin_groups:
			for E in(D.given,D.when,D.then):
				for A in E:A.enforce_candidate_limit(C)
		for A in B.teardown:A.enforce_candidate_limit(C)
class PrepPhrase(BaseModel):model_config=ConfigDict(extra=_B);preposition:str;object:str
class GrammaticalBlock(BaseModel):model_config=ConfigDict(extra=_B);order:int;subjects:List[str];verbs:List[str];past_participles:List[str];direct_objs:List[str];indirect_objs:List[str];prep_phrases:List[PrepPhrase];polarity:Literal['positive','negative'];conditions:List[str];simplified:str
class AtomicBlock(GrammaticalBlock):
	candidate_methods:List[CandidateMethod];best_candidate:CandidateMethod=Field(default_factory=lambda:CandidateMethod(declaring_class_name='',containing_class_name='',method_signature=''));notes:str
	def enforce_candidate_limit(A,limit:int=3)->_A:
		B,C=_normalize_candidates(A.candidate_methods,A.best_candidate,limit);A.candidate_methods=B
		if B:A.best_candidate=B[0]
		elif C is not _A:A.best_candidate=C
	@classmethod
	def from_grammatical_block(C,gb:GrammaticalBlock,*,candidate_methods:List[CandidateMethod]|_A=_A,best_candidate:CandidateMethod|_A=_A,notes:str='')->'AtomicBlock':B=best_candidate;A=candidate_methods;D=list(A)if A is not _A else[];E=B if B is not _A else CandidateMethod(declaring_class_name='',containing_class_name='',method_signature='');return C(**gb.model_dump(),candidate_methods=D,best_candidate=E,notes=notes)
class LocalizationEvaluationResultsOld(BaseModel):'Detailed results from the localization grader.';test_class:str;test_method:str;total_focal_methods:int;covered_focal_methods:int;uncovered_focal_methods:int;coverage_score:float;focal_methods:List[str];covered_methods:List[str];uncovered_methods:List[str];evaluation_algorithm:str;tp:int;fp:int;fn:int
class LocalizationEval(BaseModel):qualified_class_name:str;method_signature:str;all_focal_methods:List[str];covered_focal_methods:List[str];uncovered_focal_methods:List[str];tp:int;fn:int;localization_recall:float
class GrammaticalBlockList(BaseModel):grammatical_blocks:Annotated[List[GrammaticalBlock],Field(description='Ordered list of grammatical blocks.')]
class AtomicBlockList(BaseModel):
	atomic_blocks:Annotated[List[AtomicBlock],Field(description='Ordered list of atomic blocks.')]
	def enforce_candidate_limits(A,limit:int=3)->_A:
		for B in A.atomic_blocks:B.enforce_candidate_limit(limit)
class NL2LocalizationOutput(BaseModel):'Combined output from NL2Test localization evaluation.';nl2_input:NL2TestInput;localized_blocks:Union[AtomicBlockList,LocalizedScenario];evaluation_results:Optional[Union[LocalizationEvaluationResultsOld,LocalizationEval]]=_A