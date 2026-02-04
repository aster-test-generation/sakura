'\nJava code analysis utilities for NL2Test and Test2NL.\n'
_I='method_signature'
_H='same_package'
_G='same_package_or_subclass'
_F=True
_E='.'
_D='qualified_class_name'
_C='public'
_B=False
_A=None
import os,re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any,Dict,List,Literal,Optional,Set,Tuple,Union
from cldk.analysis.java import JavaAnalysis
from cldk.models.java import JCallable
from hamster.code_analysis.common import CommonAnalysis as HamsterCommonAnalysis
from hamster.code_analysis.common import Reachability as HamsterReachability
from hamster.code_analysis.common.exceptions import ClassFileNotFoundException as HamsterClassFileNotFound
from hamster.code_analysis.common.exceptions import ClassNotFoundException as HamsterClassNotFound
from hamster.code_analysis.common.exceptions import CompilationUnitNotFoundException as HamsterCompilationUnitNotFound
from hamster.code_analysis.common.exceptions import MethodNotFoundException as HamsterMethodNotFound
from hamster.code_analysis.focal_class_method.focal_class_method import FocalClassMethod
from hamster.code_analysis.model.models import TestingFramework
from hamster.code_analysis.test_statistics import SetupAnalysisInfo,TeardownAnalysisInfo
from hamster.code_analysis.utils import constants
from sakura.utils.constants import TEST_DIR
from sakura.utils.exceptions import ClassFileNotFound,ClassNotFoundError,CompilationUnitNotFound,MethodNotFoundError
from sakura.utils.pretty.prompt_formatting import pretty_indent
def _map_class_exception(qualified_class_name:str,e:Exception)->Exception:
	'Map Hamster ClassNotFoundException to Sakura ClassNotFoundError';A=qualified_class_name
	if isinstance(e,HamsterClassNotFound):return ClassNotFoundError(f"Class {A} not found (may be externally defined or misspelled).",extra_info={_D:A})
	return e
def _map_file_exception(qualified_class_name:str,e:Exception)->Exception:
	'Map Hamster file/compilation unit exceptions to Sakura exceptions';A=qualified_class_name
	if isinstance(e,HamsterClassFileNotFound):return ClassFileNotFound(f"Java file for {A} not found",extra_info={_D:A})
	if isinstance(e,HamsterCompilationUnitNotFound):return CompilationUnitNotFound(f"Compilation unit for {A} not found",extra_info={_D:A})
	return e
def _map_method_exception(qualified_class_name:str,method_signature:str,e:Exception)->Exception:
	'Map Hamster MethodNotFoundException to Sakura MethodNotFoundError';B=method_signature;A=qualified_class_name
	if isinstance(e,HamsterMethodNotFound):return MethodNotFoundError(f"Method {B} not found in class {A} (may be externally defined or misspelled).",extra_info={_D:A,_I:B})
	return e
@dataclass
class ReachabilityConfig:allow_repetition:bool=_B;only_helpers:bool=_B;add_extended_class:bool=_B
class CommonAnalysis:
	def __init__(A,analysis:JavaAnalysis):B=analysis;A.analysis=B;A._hamster=HamsterCommonAnalysis(B)
	def is_test_class(A,qualified_class_name:str,testing_frameworks:List[TestingFramework]):'\n        Determines whether a class is a test class, meaning it contains at least one test method alongside testing\n        frameworks.\n        Args:\n            qualified_class_name: The qualified class name of the class being analyzed.\n            testing_frameworks: The testing frameworks imported in the compilation unit containing the class.\n\n        Returns:\n            bool: True if the class is a test class, containing a test method, or False otherwise.\n\n        ';return A._hamster.is_test_class(qualified_class_name,testing_frameworks)
	def is_test_method(A,method_signature:str,qualified_class_name:str,testing_frameworks:List[TestingFramework])->bool:'\n        Determines whether a method, uniquely determined by its signature and qualified class name, is a test method.\n        Args:\n            method_signature: The signature of the method analyzed.\n            qualified_class_name: The qualified class name containing the method.\n            testing_frameworks: The testing frameworks imported in the compilation unit containing the class.\n\n        Returns:\n            bool: True if the method is a test method, False otherwise.\n\n        ';return A._hamster.is_test_method(method_signature,qualified_class_name,testing_frameworks,only_ascii=_F)
	def get_testing_frameworks_for_class(B,qualified_class_name:str)->List[TestingFramework]:
		"\n        Gets a list of the testing frameworks available for a class by looking at its\n        associated compilation unit and its imports.\n        Args:\n            qualified_class_name: The qualified class name of the class being analyzed.\n\n        Returns:\n            List: A list of TestingFramework objects for the class's compilation unit.\n\n        ";A=qualified_class_name
		try:return B._hamster.get_testing_frameworks_for_class(A)
		except Exception as C:raise _map_file_exception(A,C)
	def get_ncloc(A,declaration:str,body:str)->int:'\n        Get the number of non-comment lines of code.\n        Args:\n            declaration: The declaration part of the code.\n            body: The body part of the code.\n        Returns:\n            int: Number of non-comment lines.\n        ';return A._hamster.get_ncloc(declaration,body)
	def get_imports_for_class(B,qualified_class_name:str)->List[str]:
		A=qualified_class_name
		if not B.analysis.get_class(A):return[]
		C:Set[str]=set();D=B.analysis.get_java_file(qualified_class_name=A)
		if not D:raise ClassFileNotFound(f"Java file for {A} not found",extra_info={_D:A})
		E=B.analysis.get_java_compilation_unit(file_path=D)
		if not E:raise CompilationUnitNotFound(f"Compilation unit for {A} not found",extra_info={_D:A})
		for F in E.imports:C.add(F)
		return sorted(C,key=len,reverse=_F)
	def module_root_from_java_file(G,java_file:str|_A)->Path|_A:
		'Return the absolute module root inferred from a Java file path.';A=java_file
		if not A:return
		F=Path(A).expanduser().resolve();B=F.as_posix()
		for C in('/src/main/java','/src/test/java'):
			if C in B:
				D=B.split(C,1)[0].rstrip('/')
				if not D:return
				E=Path(D)
				if E==Path(_E):return
				return E
	def resolve_module_root(A,qualified_class_name:str)->Path|_A:"Resolve the absolute module root using the class's source path.";B=A.get_cldk_class_name(qualified_class_name);C=A.analysis.get_java_file(qualified_class_name=B);return A.module_root_from_java_file(C)
	def resolve_test_base_dir(D,module_root:Path|_A,*,project_root:Path|_A=_A)->Path:
		'Resolve the absolute test root directory for a module.';B=project_root;A=module_root
		if A is _A:C=Path(B).expanduser().resolve()if B is not _A else Path.cwd().resolve();return C/TEST_DIR
		return Path(A).expanduser().resolve()/TEST_DIR
	def get_referenced_app_classes(A,method_details:JCallable):
		B=set()
		for E in method_details.referenced_types:B.update(A.extract_non_parameterized_types(E))
		C=[]
		for D in B:
			if A.analysis.get_class(D)is not _A:C.append(D)
		return sorted(C,key=len)
	def get_setup_methods(B,qualified_class_name:str)->Dict[str,List[str]]:
		'\n        Returns all setup methods visible to this class, grouped by declaring class.\n        Includes inherited methods from superclasses.\n        ';A=qualified_class_name
		try:return SetupAnalysisInfo(B.analysis).get_setup_methods(A)
		except Exception as C:raise _map_file_exception(A,C)
	def get_teardown_methods(B,qualified_class_name:str)->Dict[str,List[str]]:
		'\n        Returns all teardown methods visible to this class, grouped by declaring class.\n        Includes inherited methods from superclasses.\n        ';A=qualified_class_name
		try:return TeardownAnalysisInfo(B.analysis).get_teardown_methods(A)
		except Exception as C:raise _map_file_exception(A,C)
	def get_test_methods_in_class(B,qualified_class_name:str)->List[Tuple[str,str]]:
		'\n        Returns a list of (qualified_class_name, method_signature) for all test methods in the class.\n        A method is considered a test method if is_test_method(...) evaluates to True.\n        ';A=qualified_class_name;E=B.get_testing_frameworks_for_class(A);C:List[Tuple[str,str]]=[]
		for D in B.analysis.get_methods_in_class(A):
			if B.is_test_method(D,A,E):C.append((A,D))
		return C
	def get_ascii_methods(A,qualified_class_name:str)->List[JCallable]:
		'Returns all methods in class that is ASCII';B=qualified_class_name;C:List[JCallable]=[]
		for E in A.analysis.get_methods_in_class(B):
			D=A.analysis.get_method(B,E)
			if D.code.isascii():C.append(D)
		return sorted(C,key=lambda x:len(x.signature))
	def categorize_classes(A)->Tuple[Dict[str,List[str]],List[str],List[str]]:'\n        Categorize all classes into test classes, application classes, and test utility classes.\n\n        Test utility classes are classes located in test directories (e.g., src/test/java) that\n        do not contain any test methods.\n\n        Returns:\n            Tuple containing:\n                - Dict[str, List[str]]: Mapping of test class names to their test method signatures\n                - List[str]: Application class names (production code)\n                - List[str]: Test utility class names (test helpers without test methods)\n        ';return A._hamster.categorize_classes()
	def is_subclass_of(A,sub_class:str,super_class:str)->bool:return A._hamster.is_subclass_of(sub_class,super_class)
	def implements_interface(E,class_name:str,interface_name:str)->bool:
		G=interface_name;F=class_name
		if not F or not G:return _B
		D=E.analysis.get_class(F)
		if not D:return _B
		A=[];H=set();A.extend(D.extends_list or[]);A.extend(D.implements_list or[])
		while A:
			C=A.pop()
			if C in H:continue
			if C==G:return _F
			H.add(C);B=E.analysis.get_class(C)
			if not B:continue
			if B.is_interface:A.extend(B.implements_list or[])
			else:A.extend(B.extends_list or[]);A.extend(B.implements_list or[])
		return _B
	def is_accessible_from(E,owner_class:str,method_signature:str,*,accessor_class:Optional[str]=_A,mode:Literal[_C,_H,_G]=_C)->bool:
		D=accessor_class;C=method_signature;B=owner_class
		try:return E._hamster.is_accessible_from(B,C,accessor_class=D if D else'',mode=mode)
		except Exception as A:A=_map_class_exception(B,A);A=_map_method_exception(B,C,A);raise A
	def is_public(A,qualified_class_name:str,method_signature:str)->bool:return A.is_accessible_from(qualified_class_name,method_signature,mode=_C)
	def is_abstract_class(B,qualified_class_name:str)->bool:
		'Returns True if the class is abstract.';A=B.analysis.get_class(qualified_class_name)
		if not A or not A.modifiers:return _B
		return'abstract'in A.modifiers
	def get_method_visibility(A,qualified_class_name:str,method_signature:str)->Literal[_C,_H,_G]:
		'\n        Determines the visibility level of a method.\n\n        Returns:\n            "public" if the method is accessible from anywhere\n            "same_package_or_subclass" if the method is accessible from same package or subclasses\n            "same_package" if the method is only accessible from the same package\n        ';C=method_signature;B=qualified_class_name
		if A.is_accessible_from(B,C,mode=_C):return _C
		elif A.is_accessible_from(B,C,mode=_G):return _G
		else:return _H
	def get_complicated_focal_tests(B)->Dict[str,List[str]]:
		F,I,C=B.categorize_classes();G={}
		for A in F:
			M=B.get_testing_frameworks_for_class(A);J=B.get_setup_methods(A);D=[]
			for H in F[A]:
				try:
					K=FocalClassMethod(B.analysis,I);E,C,C,C=K.extract_test_scope(A,H,J);L=len(E)>1 or len(E)==1 and len(E[0].focal_method_names)>1
					if L:D.append(H)
				except Exception:continue
			if D:G[A]=D
		return G
	def get_complicated_focal_tests_count(A)->int:B=A.get_complicated_focal_tests();return sum(len(A)for A in B.values())
	@staticmethod
	def is_getter_or_setter(method_details:JCallable)->bool:
		A=method_details
		if(A.signature.startswith('get')or A.signature.startswith('set'))and len(A.code.split('\n'))<=3:return _F
		return _B
	@staticmethod
	def get_complete_method_code(method_declaration:str,method_code:str)->str:A=method_declaration+' '+method_code;return pretty_indent(A)
	@staticmethod
	def extract_non_parameterized_types(parameterized_type:str)->List[str]:A=re.compile('[\\w\\.]+\\.[A-Z]\\w*');B=A.findall(parameterized_type);return B
	@staticmethod
	def package_of(qualified_class_name:str)->str:A=qualified_class_name;B=A.rfind(_E);return A[:B]if B!=-1 else''
	@staticmethod
	def get_simple_class_name(qualified_class_name:str)->str:A=qualified_class_name;B=A.rfind(_E);return A[B+1:]if B!=-1 else A
	@staticmethod
	def get_simple_method_name(method_signature:str)->str:A=method_signature;C=A.find('(');B=A[:C]if C!=-1 else A;D=B.rfind(_E);return B[D+1:]if D!=-1 else B
	@staticmethod
	def process_callee_signature(callee_signature:str)->str:'\n        Processes callee signature\n        Args:\n            callee_signature:\n\n        Returns:\n\n        ';A=callee_signature;D='\\b(?:[a-zA-Z_][\\w\\.]*\\.)+([a-zA-Z_][\\w]*)\\b|<[^>]*>';B=A.find('(')+1;C=A.rfind(')');E=A[B:C].split(',');F=[re.sub(D,'\\1',A.strip())for A in E];return f"{A[:B]}{', '.join(F)}{A[C:]}"
	@staticmethod
	def simplify_method_signature(method_signature:str)->str:
		'\n        Simplifies a method signature by converting fully qualified parameter types to simple names.\n        Removes generic type parameters and preserves array/varargs markers.\n\n        Used for matching method signatures when CLDK returns simple type names but\n        we need fully qualified types.\n\n        Note: Does not differentiate standard library type vs. third-party type.\n\n        Example:\n            "testMethod(com.google.common.jimfs.Configuration, java.util.List<String>)"\n            becomes "testMethod(Configuration, List)"\n        ';I='...';B=method_signature;E=B.find('(');F=B.rfind(')')
		if E==-1 or F==-1:return B
		J=B[:E];C=B[E+1:F]
		if not C.strip():return B
		while'<'in C:
			G=re.sub('<[^<>]*>','',C)
			if G==C:break
			C=G
		K=C.split(',');H=[]
		for A in K:
			A=A.strip()
			if not A:continue
			D=''
			while A.endswith('[]'):D='[]'+D;A=A[:-2]
			if A.endswith(I):D=I+D;A=A[:-3]
			if _E in A:A=A.rsplit(_E,1)[-1]
			H.append(A+D)
		return f"{J}({', '.join(H)})"
	@staticmethod
	def get_cldk_class_name(qualified_class_name:str)->str:
		'\n        Normalize inner class separators for CLDK lookups.\n        ';A=qualified_class_name
		if'$'not in A:return A
		return A.replace('$',_E)
	@staticmethod
	def get_cldk_method_sig(qualified_class_name:str,method_signature:str)->str:
		"\n        Normalize constructor signatures for CLDK lookups, since CLDK expects constructors with name '<init>'.\n        ";A=method_signature;C=qualified_class_name.split(_E)[-1];B=f"{C}("
		if B not in A:return A
		return A.replace(B,'<init>(',1)
	@staticmethod
	def normalize_path_in_project(filepath:str,project_root:Optional[str])->str:
		'\n        Normalize absolute compiler paths so that reports focus on project-relative locations.\n        ';C=project_root;B=filepath
		if not B:return B
		A=B.replace('\\','/')
		if C:
			F=str(Path(C).expanduser().resolve()).replace('\\','/');D=f"{F}/"
			if A.lower().startswith(D.lower()):return A[len(D):]
		E=A.find('/src/')
		if E!=-1:return A[E+1:]
		return os.path.basename(A)
class Reachability:
	def __init__(A,analysis:JavaAnalysis):B=analysis;A.analysis=B;A._hamster=HamsterReachability(B);(A._reachability_cache):Dict[Tuple,Dict[str,List[str]]]={}
	def get_helper_methods(A,qualified_class_name:str,method_signature:str,depth:int=constants.CONTEXT_SEARCH_DEPTH,add_extended_class:bool=_B,allow_repetition:bool=_B,only_ascii:bool=_F,test_utility_classes:List[str]|_A=_A)->Dict[str,List[str]]:'\n        Retrieves the helper methods reachable from the given method within the specified depth.\n\n        Helper methods are methods called (directly or transitively) by the given method that are\n        defined in the same class (or an extended class if add_extended_class=True).\n\n        Args:\n            qualified_class_name: The qualified name of the class.\n            method_signature: The method signature.\n            depth: The depth for search in call hierarchy.\n            add_extended_class: If set to True, include methods from classes extended by the given class.\n            allow_repetition: If set to True, allow visiting the same method multiple times in the same depth level.\n            only_ascii: If set to True, only include methods with ASCII characters.\n            test_utility_classes: List of test utility class names to include in helper method search.\n\n        Returns:\n            Dict[str, List[str]]: A map from class names to method signatures of helper methods.\n        ';return A._hamster.get_helper_methods(qualified_class_name,method_signature,depth,add_extended_class,allow_repetition,only_ascii,test_utility_classes)
	def get_concrete_classes(A,interface_class:str)->List[str]:'\n        Returns a list of concrete classes that implement the given interface class.\n\n        Args:\n            interface_class: The interface class.\n\n        Returns:\n            List[str]: List of concrete classes that implement the given interface class.\n        ';return A._hamster.get_concrete_classes(interface_class)
	def get_visible_class_methods(A,qualified_class_name:str,*,visibility_mode:Literal[_C,_H,_G]=_C,test_package:Optional[str]=_A,include_metadata:bool=_B)->Dict[str,List[Union[str,Dict[str,Any]]]]:
		'\n        Retrieves methods reachable from qualified class along its inheritance graph. Precedence looks at the class itself,\n        then superclasses, then interfaces (level-order).\n\n        Args:\n            qualified_class_name: The qualified name of the class.\n            visibility_mode: The visibility mode. Either "public", "same_package", or "same_package_or_subclass".\n            test_package: The package of the test class (for deciding whether a subclass is required).\n            include_metadata: Include metadata in the output.\n\n        Returns:\n            Dict[str, List[str]] mapping owner (class or interface) -> list of method signatures\n\n        Raises:\n            ClassNotFoundError: If the qualified_class_name cannot be found.\n        ';B=qualified_class_name;N=CommonAnalysis(A.analysis);E=A.analysis.get_class(B)
		if not E:raise ClassNotFoundError(f"Class {B} not found (may be externally defined or misspelled).",extra_info={_D:B})
		def T(owner:str,method_sig:str)->bool:return N.is_accessible_from(owner,method_sig,accessor_class=B,mode=visibility_mode)
		def U(owner:str,method_sig:str)->Dict[str,Any]:H='private';E=method_sig;D='protected';C=owner;F=A.analysis.get_method(C,E);I=N.package_of(C);B=list(F.modifiers)if F else[];G=H if H in B else _C if _C in B else D if D in B else'package-private';J=G==D and I!=test_package;return{_I:E,'declaring_qualified_class_name':C,'modifiers':B,'visibility':G,'requires_subclass':J}
		F:Dict[str,List[Union[str,Dict[str,Any]]]]={};O:set[str]=set()
		def G(owner:str)->_A:
			B=owner
			for C in A.analysis.get_methods_in_class(B):
				if C in O:continue
				if T(B,C):
					O.add(C)
					if include_metadata:F.setdefault(B,[]).append(U(B,C))
					else:F.setdefault(B,[]).append(C)
		G(B);H:deque[str]=deque(E.extends_list or[]);P:set[str]=set(E.extends_list or[]);Q:List[str]=[]
		while H:
			I=H.popleft();Q.append(I);G(I);J=A.analysis.get_class(I)
			if J and J.extends_list:
				for K in J.extends_list:
					if K not in P:P.add(K);H.append(K)
		C:deque[str]=deque();D:set[str]=set()
		def R(owner:str)->_A:
			B=A.analysis.get_class(owner)
			if B and B.implements_list:
				for E in B.implements_list:
					if E not in D:D.add(E);C.append(E)
		R(B)
		for V in Q:R(V)
		while C:
			S=C.popleft();G(S);L=A.analysis.get_class(S)
			if L and L.extends_list:
				for M in L.extends_list:
					if M not in D:D.add(M);C.append(M)
		return F
	def get_inherited_classes_and_interfaces(A,qualified_class_name:str)->List[str]:
		'\n        Returns all inherited types for the given class, first looking at superclasses then interfaces.\n        ';B=qualified_class_name;E=A.analysis.get_class(B)
		if not E:raise ClassNotFoundError(f"Class {B} not found (may be externally defined or misspelled).",extra_info={_D:B})
		F:deque[str]=deque(E.extends_list or[]);L:set[str]=set(E.extends_list or[]);G:List[str]=[]
		while F:
			M=F.popleft();G.append(M);H=A.analysis.get_class(M)
			if H and H.extends_list:
				for I in H.extends_list:
					if I not in L:L.add(I);F.append(I)
		C:deque[str]=deque();D:set[str]=set();N:List[str]=[]
		def O(owner:str)->_A:
			B=A.analysis.get_class(owner)
			if B and B.implements_list:
				for E in B.implements_list:
					if E not in D:D.add(E);C.append(E)
		O(B)
		for Q in G:O(Q)
		while C:
			P=C.popleft();N.append(P);J=A.analysis.get_class(P)
			if J and J.extends_list:
				for K in J.extends_list:
					if K not in D:D.add(K);C.append(K)
		return G+N
	def get_reachable_test_methods(A,qualified_class_name:str,testing_frameworks:List[TestingFramework])->Dict[str,List[str]]:
		"\n        Retrieves test methods reachable from the qualified class via inheritance.\n        Traverses superclasses first (BFS), then interfaces.\n\n        Args:\n            qualified_class_name: The qualified name of the class.\n            testing_frameworks: Testing frameworks available in this class's compilation unit.\n\n        Returns:\n            Dict[str, List[str]] mapping declaring class -> list of test method signatures\n\n        Raises:\n            ClassNotFoundError: If the qualified_class_name cannot be found.\n        ";B=qualified_class_name;G=CommonAnalysis(A.analysis);C=A.analysis.get_class(B)
		if not C:raise ClassNotFoundError(f"Class {B} not found (may be externally defined or misspelled).",extra_info={_D:B})
		def M(owner:str,method_sig:str)->bool:
			A=owner;B=G.get_testing_frameworks_for_class(A)
			if not B:return _B
			return G.is_test_method(method_sig,A,B)
		H:Dict[str,List[str]]={};I:set[str]=set()
		def J(owner:str)->_A:
			C=owner
			for B in A.analysis.get_methods_in_class(C):
				if B in I:continue
				if M(C,B):I.add(B);H.setdefault(C,[]).append(B)
		J(B);D:deque[str]=deque(C.extends_list or[]);K:set[str]=set(C.extends_list or[])
		while D:
			L=D.popleft();J(L);E=A.analysis.get_class(L)
			if E and E.extends_list:
				for F in E.extends_list:
					if F not in K:K.add(F);D.append(F)
		return H