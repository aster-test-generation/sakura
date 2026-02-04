from __future__ import annotations
_A=None
from functools import singledispatchmethod
from pathlib import Path
from typing import Dict,Iterable,List,Sequence,Set,Tuple
from cldk.analysis.java import JavaAnalysis
from hamster.code_analysis.focal_class_method.focal_class_method import FocalClassMethod
from hamster.code_analysis.test_statistics import SetupAnalysisInfo
from sakura.nl2test.models import AgentState,AtomicBlockList,LocalizedScenario,NL2TestInput
from sakura.nl2test.models.decomposition import DecompositionMode,LocalizationEval,LocalizedStep
from sakura.utils.analysis import CommonAnalysis
from sakura.utils.pretty.color_logger import RichLog
from sakura.utils.pretty.prints import pretty_print
FocalMethod=Tuple[str,str]
SemanticKey=Tuple[str,str]
class LocalizationGrader:
	def __init__(A,analysis:JavaAnalysis,project_root:Path,decomposition_mode:DecompositionMode,application_classes:Sequence[str],test_utility_classes:Sequence[str]|_A=_A)->_A:B=analysis;A.analysis=B;A.project_root=Path(project_root);A.decomposition_mode=decomposition_mode;A.application_classes=list(application_classes);(A.test_utility_classes):List[str]=list(test_utility_classes or[]);A.common_analysis=CommonAnalysis(B)
	def set_analysis(A,analysis:JavaAnalysis)->_A:'Update the underlying analysis used for grading.';B=analysis;A.analysis=B;A.common_analysis=CommonAnalysis(B)
	@singledispatchmethod
	def grade(self,obj,nl2_input:NL2TestInput)->LocalizationEval:raise TypeError('Unsupported input type for grade().')
	@grade.register
	def _(self,atomic_blocks:AtomicBlockList,nl2_input:NL2TestInput)->LocalizationEval:raise NotImplementedError('Grading for AtomicBlockList is not implemented in the new grader.')
	@grade.register
	def _(self,scenario:LocalizedScenario,nl2_input:NL2TestInput)->LocalizationEval:
		B=nl2_input;A=self;C=A._get_focal_methods(B)
		if not C:return A._empty_results(B)
		D=A._collect_covered_methods(scenario,C);G=C-D;E=len(D);H=len(G);I=E+H;F=E/I if I else 1.;F=round(F,4);return LocalizationEval(qualified_class_name=B.qualified_class_name,method_signature=B.method_signature,all_focal_methods=A._format_methods(C),covered_focal_methods=A._format_methods(D),uncovered_focal_methods=A._format_methods(G),tp=E,fn=H,localization_recall=F)
	def grade_from_state(B,supervisor_state:AgentState|_A,nl2_input:NL2TestInput)->LocalizationEval:
		D=supervisor_state;A=nl2_input;E=B._get_focal_methods(A)
		if not E:return B._empty_results(A)
		C=_A
		if D is not _A:
			if B.decomposition_mode==DecompositionMode.GHERKIN:C=D.localized_scenario
			else:C=D.atomic_blocks
		if C is _A:return B._missing_output_results(A,E)
		try:return B.grade(C,A)
		except Exception as F:RichLog.warn(f"Localization evaluation failed for {A.qualified_class_name}::{A.method_signature} (id={A.id}): {F}");return B._missing_output_results(A,E)
	def _empty_results(B,nl2_input:NL2TestInput)->LocalizationEval:A=nl2_input;return LocalizationEval(qualified_class_name=A.qualified_class_name,method_signature=A.method_signature,all_focal_methods=[],covered_focal_methods=[],uncovered_focal_methods=[],tp=0,fn=0,localization_recall=round(1.,4))
	def _missing_output_results(B,nl2_input:NL2TestInput,focal_methods:Set[FocalMethod])->LocalizationEval:C=nl2_input;A=focal_methods;return LocalizationEval(qualified_class_name=C.qualified_class_name,method_signature=C.method_signature,all_focal_methods=B._format_methods(A),covered_focal_methods=[],uncovered_focal_methods=B._format_methods(A),tp=0,fn=len(A),localization_recall=.0)
	def _get_focal_methods(B,nl2_input:NL2TestInput)->Set[FocalMethod]:
		A=nl2_input
		try:
			F:Dict[str,List[str]]=SetupAnalysisInfo(B.analysis).get_setup_methods(A.qualified_class_name);G=FocalClassMethod(B.analysis,B.application_classes,B.test_utility_classes);H,C,C,C=G.extract_test_scope(A.qualified_class_name,A.method_signature,F);D:Set[FocalMethod]=set()
			for E in H:
				for I in E.focal_method_names:D.add((E.focal_class,I))
			return D
		except Exception as J:pretty_print('Error getting focal methods',{'test_class':A.qualified_class_name,'test_method':A.method_signature,'error':str(J)});return set()
	def _get_semantic_key(C,qualified_class_name:str,method_signature:str)->SemanticKey:'\n        Extract a relaxed matching key from a focal method.\n        ';A=qualified_class_name.split('.')[-1].split('$')[-1];B=method_signature.split('(')[0].strip();return A,B
	def _collect_covered_methods(A,scenario:LocalizedScenario,focal_methods:Set[FocalMethod])->Set[FocalMethod]:
		'\n        Identifies which ground truth focal methods were predicted by the model.\n        ';C:Set[FocalMethod]=set();B:Dict[SemanticKey,Set[FocalMethod]]={}
		for(D,E)in focal_methods:G=A._get_semantic_key(D,E);B.setdefault(G,set()).add((D,E))
		for H in A._iter_steps(scenario):
			I=A._collect_candidates(H)
			for(J,K)in I:
				F=A._get_semantic_key(J,K)
				if F in B:C.update(B[F])
		return C
	def _collect_candidates(C,step:LocalizedStep)->Set[FocalMethod]:
		B:Set[FocalMethod]=set()
		for A in step.candidate_methods or[]:
			if A.containing_class_name and A.method_signature:B.add((A.containing_class_name,A.method_signature))
		return B
	def _iter_steps(C,scenario:LocalizedScenario)->Iterable[LocalizedStep]:
		A=scenario;yield from A.setup or[]
		for B in A.gherkin_groups or[]:yield from B.given or[];yield from B.when or[];yield from B.then or[]
		yield from A.teardown or[]
	def _format_methods(A,methods:Set[FocalMethod])->List[str]:return sorted(f"{A}.{B}"for(A,B)in methods)