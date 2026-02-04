_B=False
_A=None
from typing import List,Set
from cldk.analysis.java import JavaAnalysis
from hamster.code_analysis.test_statistics import CallAndAssertionSequenceDetailsInfo
from sakura.test2nl.model.models import CallSiteInfo,MethodContext,VariableInfo
from sakura.utils.analysis import CommonAnalysis
class MethodExtractor:
	def __init__(A,analysis:JavaAnalysis,application_classes:List[str]):A.analysis=analysis;A.application_classes=application_classes
	def _is_helper_class(A,type_name:str|_A)->bool|_A:
		'Check if type exists in repo but is not an application class.';B=type_name
		if not B:return
		D=CommonAnalysis.extract_non_parameterized_types(B)
		for C in D:
			if A.analysis.get_class(C)and C not in A.application_classes:return True
		return _B
	def extract(D,qualified_class_name:str,method_signature:str,complete_methods:bool,include_class_name:bool=_B)->MethodContext:
		F=method_signature;E=qualified_class_name;A=D.analysis.get_method(E,F)
		if not A:raise ValueError(f"Method {F} not found in class {E}")
		if CommonAnalysis.is_getter_or_setter(A):G=_A;H=True
		elif complete_methods:N=CommonAnalysis.get_complete_method_code(A.declaration,A.code);G=N.strip();H=_B
		else:G=_A;H=_B
		J:Set[str]=set()
		try:
			O=CommonAnalysis(D.analysis);P=O.get_testing_frameworks_for_class(E);Q=CallAndAssertionSequenceDetailsInfo(D.analysis);R=Q.get_call_and_assertion_sequence_details_info(E,F,P)
			for S in R:
				for T in S.assertion_details:J.add(T.assertion_name)
		except Exception:pass
		K:list[CallSiteInfo]=[]
		for B in A.call_sites or[]:U=B.method_name in J;V=D._is_helper_class(B.receiver_type);K.append(CallSiteInfo(method_name=B.method_name,receiver_type=B.receiver_type,return_type=B.return_type,line_number=B.start_line,is_assertion=U,is_helper=V))
		L:list[VariableInfo]=[]
		for C in A.variable_declarations or[]:W=D._is_helper_class(C.type);L.append(VariableInfo(name=C.name,type=C.type,initializer=C.initializer if C.initializer else _A,line_number=C.start_line,type_is_helper_class=W))
		M:str|_A=_A
		for I in A.comments or[]:
			if I.is_javadoc and I.content:M=I.content;break
		return MethodContext(method_signature=A.signature,qualified_class_name=E if include_class_name else _A,is_getter_or_setter=H,code=G,call_sites=K,variable_declarations=L,thrown_exceptions=A.thrown_exceptions or[],javadoc=M)