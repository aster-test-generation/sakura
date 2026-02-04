from __future__ import annotations
_C='get_call_site_details'
_B='extract_method_code'
_A='get_method_details'
from typing import Any,Dict,Set
from sakura.utils.analysis.java_analyzer import CommonAnalysis
class CLDKArgNormalizer:
	'\n    Shared utility for normalizing tool arguments for CLDK compatibility.\n    Handles inner class name normalization and constructor method signature normalization.\n    ';NORMALIZE_CLASS_TOOLS:Set[str]={_A,'get_class_fields','get_class_imports','get_class_constructors_and_factories','get_getters_and_setters',_B,_C,'search_reachable_methods_in_class','get_class_details','get_inherited_library_classes'};NORMALIZE_METHOD_SIG_TOOLS:Set[str]={_C,_A,_B}
	@staticmethod
	def normalize_args(tool_name:str,raw_args:Dict[str,Any])->Dict[str,Any]:
		'\n        Normalize tool arguments for CLDK compatibility.\n\n        Handles:\n        - Inner class name normalization (e.g., Outer$Inner -> Outer.Inner for CLDK)\n        - Constructor method signature normalization\n\n        Args:\n            tool_name: The name of the tool being called\n            raw_args: The original tool arguments\n\n        Returns:\n            Normalized arguments dict (may be same object if no changes needed)\n        ';I='method_signature';F=tool_name;E='qualified_class_name';C=raw_args;A:Dict[str,Any]=C
		if F in CLDKArgNormalizer.NORMALIZE_CLASS_TOOLS:
			B=C.get(E)
			if isinstance(B,str):
				G=CommonAnalysis.get_cldk_class_name(B)
				if G!=B:
					if A is C:A=dict(A)
					A[E]=G
		if F in CLDKArgNormalizer.NORMALIZE_METHOD_SIG_TOOLS:
			B=A.get(E);D=A.get(I)
			if B and D:
				H=CommonAnalysis.get_cldk_method_sig(B,D)
				if H!=D:
					if A is C:A=dict(A)
					A[I]=H
		return A