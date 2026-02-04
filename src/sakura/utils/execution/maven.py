_I='classname'
_H='testcase'
_G='testsuites'
_F='message'
_E=False
_D='testsuite'
_C='failure'
_B='error'
_A=None
import os,re,time,xml.etree.ElementTree as ET
from pathlib import Path
from typing import List,Optional,Tuple
from javabuild.maven_build import MavenBuild
from pydantic import BaseModel
from sakura.utils.analysis import CommonAnalysis
class ExecutionIssue(BaseModel):'Single failure or error captured from a Maven test run.';class_name:str;test_name:str;kind:str;message:Optional[str]=_A;error_type:Optional[str]=_A;file:Optional[str]=_A;line:Optional[int]=_A;stack_trace:Optional[str]=_A
class JavaMavenExecution(MavenBuild):
	'Run Maven tests and parse failures/errors from Surefire or console logs.';HEAD_STACKTRACE_LINES=8;TAIL_STACKTRACE_LINES=4;FAILURE_HEADER_RE=re.compile('^\\[ERROR\\]\\s+(?:(?P<class>[A-Za-z0-9_.$]+)\\.(?P<test>[^\\s]+)|(?P<test2>[^\\s(]+)\\((?P<class2>[A-Za-z0-9_.$]+)\\))\\s+.*<<<\\s+(?P<kind>FAILURE|ERROR)!');FAILURE_SUMMARY_RE=re.compile('^\\[ERROR\\]\\s+(?P<class>[A-Za-z0-9_.$]+)\\.(?P<test>[^:]+?)(?::(?P<line>\\d+))?\\s+(?P<message>.+)$');STACK_FILE_RE=re.compile('\\(([^()/:\\\\\\s]+\\.java):(\\d+)\\)');STACK_FALLBACK_FILE_RE=re.compile('([A-Za-z0-9_\\-./\\\\]+\\.java):(\\d+)')
	def __init__(F,project_root:Path,module_root:Path|_A=_A):
		D=module_root;A=project_root.resolve();B=D.resolve()if D is not _A else _A;C=_A
		if B and A!=B:
			try:
				E=B.relative_to(A).as_posix()
				if E not in('','.'):C=E
			except ValueError:C=_A
		if C:super().__init__(str(A),target_module=C)
		else:super().__init__(str(A))
		F.module_root=B or A
	def get_execution_errors(A,qualified_class_name:Optional[str]=_A,method_signature:Optional[str]=_A,prepare_dependencies:bool=_E,timeout:int|_A=600)->List[ExecutionIssue]:
		G=timeout;F=method_signature;E=qualified_class_name
		if prepare_dependencies and A.target_module:A.install_selected_projects_skip_tests_proc(timeout=G)
		B:Optional[str]=_A;C:Optional[str]=E;D:Optional[str]=_A
		if E:
			H=CommonAnalysis.get_simple_class_name(E)
			if F:D=CommonAnalysis.get_simple_method_name(F);B=f"{H}#{D}"
			else:B=H
		K=time.time()-2.;L=A.run_tests_proc(target_tests=B,also_make=_E,timeout=G);M=L.stdout;I=A.find_surefire_reports()
		if I and C:
			N,O=A.parse_surefire_reports_for_target(I,min_mtime=K,expected_class=C,expected_method=D)
			if O:return N
		J=A.parse_console_output(M)
		if J:return J
		if C and B:return[ExecutionIssue(class_name=C,test_name=D or'',kind=_B,error_type='NoSpecifiedTests',message='Specified test was not executed (no matching surefire testcase found).')]
		return[]
	def find_surefire_reports(G)->List[str]:
		'Locate every target/surefire-reports directory in the project tree.';F='target';A='surefire-reports';B:List[str]=[];C=G.module_root;D=C/F/A
		if D.is_dir():return[str(D)]
		for(E,H,I)in os.walk(str(C)):
			if A in H and os.path.basename(E)==F:B.append(os.path.join(E,A))
		return B
	def parse_surefire_reports_for_target(A,reports_dirs:List[str],min_mtime:Optional[float]=_A,expected_class:Optional[str]=_A,expected_method:Optional[str]=_A)->Tuple[List[ExecutionIssue],bool]:
		I=expected_class;H=min_mtime;D=expected_method;J:List[ExecutionIssue]=[];K=_E
		for L in reports_dirs:
			try:R=[A for A in os.listdir(L)if A.startswith('TEST-')and A.endswith('.xml')]
			except FileNotFoundError:continue
			for S in R:
				M=os.path.join(L,S)
				if H is not _A:
					try:
						if os.path.getmtime(M)<H:continue
					except OSError:pass
				try:T=ET.parse(M);B=T.getroot()
				except Exception:continue
				C:List[ET.Element]=[];N=A._local_name(B.tag)
				if N==_D:C=[B]
				elif N==_G:C=[B for B in B if A._local_name(B.tag)==_D]
				else:C=[B for B in B.iter()if A._local_name(B.tag)==_D]
				for U in C:
					for E in(B for B in U if A._local_name(B.tag)==_H):
						O=E.get(_I)or'';F=E.get('name')or''
						if I and O!=I:continue
						if D:
							if not(F==D or F.startswith(D)):continue
						K=True
						for P in(_C,_B):
							for G in(B for B in E if A._local_name(B.tag)==P):Q=ExecutionIssue(class_name=O,test_name=F,kind=_C if P==_C else _B,message=G.get(_F),error_type=G.get('type'),stack_trace=A._truncate_stack_trace((G.text or'').strip()or _A));A._maybe_fill_file_line(Q);J.append(Q)
		return J,K
	def parse_surefire_reports(A,reports_dirs:List[str],min_mtime:Optional[float]=_A)->List[ExecutionIssue]:
		'Read Surefire XML reports and convert every failure/error into ExecutionIssue objects.';F=min_mtime;G:List[ExecutionIssue]=[]
		for H in reports_dirs:
			try:M=[A for A in os.listdir(H)if A.startswith('TEST-')and A.endswith('.xml')]
			except FileNotFoundError:continue
			for N in M:
				I=os.path.join(H,N)
				if F is not _A:
					try:
						if os.path.getmtime(I)<F:continue
					except OSError:pass
				try:O=ET.parse(I);B=O.getroot()
				except Exception:continue
				C:List[ET.Element]=[];J=A._local_name(B.tag)
				if J==_D:C=[B]
				elif J==_G:C=[B for B in B if A._local_name(B.tag)==_D]
				else:C=[B for B in B.iter()if A._local_name(B.tag)==_D]
				for P in C:
					for D in(B for B in P if A._local_name(B.tag)==_H):
						Q=D.get(_I)or'';R=D.get('name')or''
						for K in(_C,_B):
							for E in(B for B in D if A._local_name(B.tag)==K):L=ExecutionIssue(class_name=Q,test_name=R,kind=_C if K==_C else _B,message=E.get(_F),error_type=E.get('type'),stack_trace=A._truncate_stack_trace((E.text or'').strip()or _A));A._maybe_fill_file_line(L);G.append(L)
		return G
	def parse_console_output(B,stdout:Optional[str])->List[ExecutionIssue]:
		'Best-effort parsing of Maven console logs when Surefire XML is unavailable.';T='test';S='class';H=stdout
		if not H:return[]
		I:List[ExecutionIssue]=[];F=H.splitlines();C=0
		while C<len(F):
			J=F[C];A=B.FAILURE_HEADER_RE.match(J);D=_A if A else B.FAILURE_SUMMARY_RE.match(J)
			if A or D:
				if A:K=(A.group(S)or A.group('class2')).strip();L=(A.group(T)or A.group('test2')).strip();G=_C if A.group('kind').upper()=='FAILURE'else _B;E=_A;M=_A
				else:
					K=D.group(S);L=D.group(T).strip();N=D.group('line');M=int(N)if N else _A;E=D.group(_F).strip();O=E.lower()
					if _B in O or'exception'in O:G=_B
					else:G=_C
				P,U=B._collect_stack_trace(F,C+1);V,Q=B._extract_error_type_and_message(P)
				if not E and Q:E=Q
				R=ExecutionIssue(class_name=K,test_name=L,kind=G,message=E,error_type=V,line=M,stack_trace=P);B._maybe_fill_file_line(R);I.append(R);C=U;continue
			C+=1
		return I
	@classmethod
	def _collect_stack_trace(F,lines:List[str],start_index:int)->Tuple[Optional[str],int]:
		K='[INFO]';J='[ERROR]';I=lines;C:List[str]=[];D:int=start_index;G=0
		while D<len(I):
			B:str=I[D]
			if F.FAILURE_HEADER_RE.match(B)or F.FAILURE_SUMMARY_RE.match(B):break
			A:str=B.strip()
			if A.startswith('[INFO] Results')or A.startswith('[INFO] Tests run')or A.startswith('[ERROR] Results')or A.startswith('[ERROR] Tests run')or'There are test failures'in A:break
			if A.startswith('[INFO] BUILD')or A.startswith('[ERROR] BUILD')or A.startswith('[INFO] ---')or A.startswith('[ERROR] ---'):break
			E=B
			if B.startswith(J):E=B[len(J):].lstrip()
			elif B.startswith(K):E=B[len(K):].lstrip()
			if not E:
				G+=1
				if G>1 and C:D+=1;break
			else:G=0
			if E or C:C.append(E)
			D+=1
		H='\n'.join(C).strip()if C else _A;H=F._truncate_stack_trace(H);return H,D
	@classmethod
	def _maybe_fill_file_line(C,issue:ExecutionIssue)->_A:
		A=issue
		if A.line is not _A and A.file:return
		if not A.stack_trace:return
		B=C.STACK_FILE_RE.search(A.stack_trace)
		if not B:B=C.STACK_FALLBACK_FILE_RE.search(A.stack_trace)
		if B:
			A.file=B.group(1)
			try:A.line=int(B.group(2))
			except ValueError:A.line=_A
	@classmethod
	def _truncate_stack_trace(A,text:Optional[str])->Optional[str]:
		B=text
		if not B:return
		C=B.splitlines();D=len(C);E=A.HEAD_STACKTRACE_LINES+A.TAIL_STACKTRACE_LINES
		if D<=E:return B
		F=D-E;G=C[:A.HEAD_STACKTRACE_LINES];H=C[-A.TAIL_STACKTRACE_LINES:];return'\n'.join(G)+f"\n    ... (skipped {F} lines) ...\n"+'\n'.join(H)
	@staticmethod
	def _extract_error_type_and_message(stack_trace:Optional[str])->Tuple[Optional[str],Optional[str]]:
		C=stack_trace
		if not C:return _A,_A
		B=C.splitlines()[0].strip()
		if not B:return _A,_A
		A,D,E=B.partition(':')
		if D and('.'in A or A.endswith('Error')or A.endswith('Exception')):F=A.strip();G=E.strip()or _A;return F,G
		return _A,B
	@staticmethod
	def _local_name(tag:str)->str:A=tag;return A.split('}',1)[-1]if'}'in A else A