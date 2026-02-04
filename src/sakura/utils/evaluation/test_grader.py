from __future__ import annotations
_C=None
_B=1.
_A=.0
from pathlib import Path
from typing import Any,Dict,List,Optional,Set,Tuple
from cldk.analysis.java import JavaAnalysis
from cldk.models.java import JCallable
from hamster.code_analysis.focal_class_method.focal_class_method import FocalClassMethod
from hamster.code_analysis.model.models import AssertionDetails,AssertionType,CallableDetails,CallAndAssertionSequenceDetails
from hamster.code_analysis.test_statistics import SetupAnalysisInfo,TestMethodAnalysisInfo
from sakura.utils.analysis import CommonAnalysis
from sakura.utils.coverage.individual_test_coverage import IndividualTestCoverage
from sakura.utils.models import NL2TestCoverageEval,NL2TestInput,NL2TestMetadata,NL2TestStructuralEval
from sakura.utils.pretty.color_logger import RichLog
class TestGrader:
	'\n    Grader for NL2Test structural and coverage evaluation.\n    '
	def __init__(A,analysis:JavaAnalysis,project_root:Path,application_classes:Optional[List[str]]=_C,test_utility_classes:Optional[List[str]]=_C)->_C:B=analysis;A.analysis=B;A.project_root=project_root;(A.application_classes):List[str]=list(application_classes or[]);(A.test_utility_classes):List[str]=list(test_utility_classes or[]);A.common=CommonAnalysis(B)
	def set_analysis(A,analysis:JavaAnalysis)->_C:B=analysis;A.analysis=B;A.common=CommonAnalysis(B)
	@staticmethod
	def _count_num_assertions(seqs:List[CallAndAssertionSequenceDetails])->int:return sum(len(A.assertion_details)for A in seqs)
	@staticmethod
	def _get_assertion_types(seqs:List[CallAndAssertionSequenceDetails])->List[List[AssertionType]]:
		A:List[List[AssertionType]]=[]
		for B in seqs:
			for C in B.assertion_details:A.append(C.assertion_type)
		return A
	@staticmethod
	def _get_expanded_seqs(seqs:List[CallAndAssertionSequenceDetails])->List[CallableDetails|AssertionDetails]:
		A:List[CallableDetails|AssertionDetails]=[]
		for B in seqs:A.extend(B.call_sequence_details);A.extend(B.assertion_details)
		return A
	@staticmethod
	def _assertion_scores(gt_assertions:List[List[AssertionType]],pred_assertions:List[List[AssertionType]])->Tuple[float,float]:
		'Compute recall and precision over assertion groups.';A=[set(A or[])for A in gt_assertions];B=[set(A or[])for A in pred_assertions];D:Set[int]=set();C:Set[int]=set()
		for(F,G)in enumerate(A):
			for(E,H)in enumerate(B):
				if E in C:continue
				if any(A in H for A in G):D.add(F);C.add(E);break
		I=len(D)/len(A)if A else _B;J=len(C)/len(B)if B else _B;return I,J
	@staticmethod
	def _callable_scores(gt_seqs:List[CallAndAssertionSequenceDetails],pred_seqs:List[CallAndAssertionSequenceDetails])->Tuple[float,float]:
		'Compute recall and precision for callables and assertions combined.';from collections import Counter as A;B=TestGrader._get_expanded_seqs(gt_seqs);C=TestGrader._get_expanded_seqs(pred_seqs);D=A((A.receiver_type,A.method_name)for A in B if isinstance(A,CallableDetails));E=A((A.receiver_type,A.method_name)for A in C if isinstance(A,CallableDetails));N=[A for A in B if isinstance(A,AssertionDetails)];O=[A for A in C if isinstance(A,AssertionDetails)];F=[set(A.assertion_type or[])for A in N];G=[set(A.assertion_type or[])for A in O];H=0;I:Set[int]=set()
		for P in F:
			for(J,Q)in enumerate(G):
				if J in I:continue
				if any(A in Q for A in P):H+=1;I.add(J);break
		R=(D&E).total();K=sum(D.values())+len(F);L=sum(E.values())+len(G);M=R+H;S=M/K if K else _B;T=M/L if L else _B;return S,T
	@staticmethod
	def _zero_structural_eval()->NL2TestStructuralEval:return NL2TestStructuralEval(obj_creation_recall=_A,obj_creation_precision=_A,assertion_recall=_A,assertion_precision=_A,callable_recall=_A,callable_precision=_A,focal_recall=_A,focal_precision=_A)
	@staticmethod
	def _zero_coverage_eval()->NL2TestCoverageEval:return NL2TestCoverageEval(class_coverage=_A,method_coverage=_A,line_coverage=_A,branch_coverage=_A)
	def _get_constructor_types(E,class_name:str,method_sig:str)->Set[str]:
		A:Set[str]=set()
		try:B:Optional[JCallable]=E.analysis.get_method(qualified_class_name=class_name,qualified_method_name=method_sig)
		except Exception:B=_C
		if not B:return A
		for C in B.call_sites:
			if C.is_constructor_call:
				D=C.return_type
				if D:A.add(D)
		return A
	def _get_focal_methods_for_test(A,qualified_class_name:str,method_signature:str)->Set[Tuple[str,str]]:
		'Collect focal (class, method) pairs for a given test method.';B=qualified_class_name
		try:
			F=SetupAnalysisInfo(A.analysis).get_setup_methods(B);G=FocalClassMethod(A.analysis,A.application_classes,A.test_utility_classes);H,C,C,C=G.extract_test_scope(B,method_signature,F);D:Set[Tuple[str,str]]=set()
			for E in H:
				for I in E.focal_method_names:D.add((E.focal_class,I))
			return D
		except Exception:return set()
	def grade_structural(A,pred_method_sig:str,pred_class_name:str,gt_method_sig:str,gt_class_name:str)->Optional[NL2TestStructuralEval]:
		N='TestDataset';E=gt_method_sig;D=pred_method_sig;C=gt_class_name;B=pred_class_name
		if not A.analysis.get_class(B)or not A.analysis.get_method(B,D):return
		O=A.common.get_testing_frameworks_for_class(C);P=A.common.get_testing_frameworks_for_class(B);Q:Dict[str,List[str]]=SetupAnalysisInfo(A.analysis).get_setup_methods(C);J=TestMethodAnalysisInfo(A.analysis,N,A.application_classes).get_test_method_analysis_info(O,C,E,Q);R:Dict[str,List[str]]=SetupAnalysisInfo(A.analysis).get_setup_methods(B);K=TestMethodAnalysisInfo(A.analysis,N,A.application_classes).get_test_method_analysis_info(P,B,D,R);F=A._get_constructor_types(C,E);G=A._get_constructor_types(B,D);L=F&G;S=len(L)/len(F)if F else _B;T=len(L)/len(G)if G else _B;U=A._get_assertion_types(J.call_assertion_sequences or[]);V=A._get_assertion_types(K.call_assertion_sequences or[]);W,X=A._assertion_scores(U,V);Y,Z=A._callable_scores(J.call_assertion_sequences or[],K.call_assertion_sequences or[]);H=A._get_focal_methods_for_test(C,E);I=A._get_focal_methods_for_test(B,D);M=H&I;a=len(M)/len(H)if H else _B;b=len(M)/len(I)if I else _B;return NL2TestStructuralEval(obj_creation_recall=round(S,4),obj_creation_precision=round(T,4),assertion_recall=round(W,4),assertion_precision=round(X,4),callable_recall=round(Y,4),callable_precision=round(Z,4),focal_recall=round(a,4),focal_precision=round(b,4))
	def grade_coverage(B,pred_method_sig:str,pred_class_name:str,gt_method_sig:str,gt_class_name:str)->Optional[NL2TestCoverageEval]:
		'\n        Compute coverage overlap between prediction and ground truth.\n        Returns per-metric fractions in [0.0, 1.0] (non-penalizing 1.0 when GT is empty).\n        ';V='coverage_details';U='test_name';I=gt_method_sig;G='(';D=pred_method_sig;C=gt_class_name;A=pred_class_name
		if not B.analysis.get_class(A)or not B.analysis.get_method(A,D):return
		W=[(A,D.split(G)[0]),(C,I.split(G)[0])];J=B.common.resolve_module_root(C)or B.common.resolve_module_root(A)
		try:X=IndividualTestCoverage.from_common_analysis(B.common,project_root=B.project_root,qualified_class_names=[C,A],module_root=J);E=X.generate(tests_to_run=W)
		except Exception as K:
			if J is _C:RichLog.warn(f"Coverage evaluation failed because module root could not be resolved for {C} or {A}: {K}")
			else:RichLog.warn(f"Coverage evaluation failed for {A}::{D}: {K}")
			return
		L:Dict[str,Dict[str,Any]]={};M:Dict[str,Dict[str,Any]]={}
		if C in E:
			for N in E[C]:
				if N[U]==I.split(G)[0]:L=N[V]
		if A in E:
			for O in E[A]:
				if O[U]==D.split(G)[0]:M=O[V]
		def H(x)->Set[int]:
			if x is _C:return set()
			if isinstance(x,set):return x
			if isinstance(x,(list,tuple)):return set(x)
			return{x}
		def P(coverage_map:Dict[str,Dict[str,Any]])->Tuple[Set[str],Set[Tuple[str,str]],Set[Tuple[str,int]],Set[Tuple[str,int]]]:
			F:Set[str]=set();G:Set[Tuple[str,str]]=set();I:Set[Tuple[str,int]]=set();J:Set[Tuple[str,int]]=set()
			for(B,A)in(coverage_map or{}).items():
				if not isinstance(A,dict):continue
				for L in A.get('method_covered',[])or[]:G.add((B,L))
				K=H(A.get('covered_lines'));C=H(A.get('branch_lines_covered'));D=H(A.get('branch_lines_partial'))
				if K or C or D:F.add(B)
				for E in K|C|D:I.add((B,E))
				for E in C|D:J.add((B,E))
			return F,G,I,J
		Q,R,S,T=P(L);Y,Z,a,b=P(M);c=Y&Q;d=Z&R;e=a&S;f=b&T
		def F(n:int,d:int)->float:return n/d if d>0 else _B
		return NL2TestCoverageEval(class_coverage=round(F(len(c),len(Q)),4),method_coverage=round(F(len(d),len(R)),4),line_coverage=round(F(len(e),len(S)),4),branch_coverage=round(F(len(f),len(T)),4))
	def grade(A,nl2_input:NL2TestInput,nl2_metadata:NL2TestMetadata,*,compiles:bool=True)->Tuple[Optional[NL2TestStructuralEval],Optional[NL2TestCoverageEval]]:
		O='signature';J=nl2_metadata;I=nl2_input
		if not compiles:return A._zero_structural_eval(),A._zero_coverage_eval()
		C=J.qualified_test_class_name
		if not A.analysis.get_class(C):return A._zero_structural_eval(),A._zero_coverage_eval()
		E=(J.method_signature or'').strip()
		def F(sig:str)->bool:
			B=False
			if not sig:return B
			try:return bool(A.analysis.get_method(C,sig))
			except Exception:return B
		if not F(E):
			G=A.common.get_test_methods_in_class(C);B=''
			if G:
				for(R,D)in G:
					if F(D):B=D;break
				else:B=G[0][1]
			if not B:
				H=list(A.analysis.get_methods_in_class(C)or[])
				for K in H:
					D=getattr(K,O,_C)or str(K)
					if F(D):B=D;break
				if not B and H:L=H[0];B=getattr(L,O,L)
			E=B.strip()
		M=I.qualified_class_name;N=I.method_signature;P=A.grade_structural(pred_method_sig=E,pred_class_name=C,gt_method_sig=N,gt_class_name=M);Q=A.grade_coverage(pred_method_sig=E,pred_class_name=C,gt_method_sig=N,gt_class_name=M);return P,Q