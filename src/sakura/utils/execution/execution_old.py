_F='SurefireBooterForkException'
_E='BUILD FAILURE'
_D='BUILD_ERROR'
_C=True
_B=False
_A=None
import os,re,sys,shutil,subprocess,xml.etree.ElementTree as ET
from typing import Optional,List
from pydantic import BaseModel,Field
from sakura.utils.constants import MAVEN_CMD
MAX_STACKTRACE_LINES_ON_FAILURE=12
class TestIssue(BaseModel):'Single failure or error in a test case.';class_name:str;test_name:str;kind:str;message:Optional[str]=_A;error_type:Optional[str]=_A;time_sec:Optional[float]=_A;file:Optional[str]=_A;line:Optional[int]=_A;stack_trace:Optional[str]=_A
class TestExecutionFeedback(BaseModel):'Structured result of a Maven/Surefire test run, with minimized logs.';command:str;returncode:int;test_class_name:Optional[str]=_A;timed_out:bool=_B;passed:bool;tests_run:int;failures:int;errors:int;skipped:int;time_elapsed_sec:float;issues:List[TestIssue]=Field(default_factory=list);stdout_summary:Optional[str]=_A;stderr:Optional[str]=_A;reports_dirs:List[str]=Field(default_factory=list);failure_reason_code:Optional[str]=_A;failure_reason:Optional[str]=_A
class JavaExecution:
	'\n    Execute tests in a Maven-based Java project and return parsed feedback.\n    Assumes projects are Maven and pre-compiled; runs only the test phase.\n    '
	@staticmethod
	def execute(project_root:str,test_class_name:Optional[str],timeout:int=900)->TestExecutionFeedback:
		K=' ';G=project_root;C=test_class_name
		if not C:raise ValueError('test_class_name is required for this executor.')
		O=os.path.join(G,'pom.xml')
		def P(project_root:str)->List[str]:
			E='win32';B=project_root;C='mvnw.cmd'if sys.platform==E else'mvnw';A=os.path.join(B,C)
			if os.path.isfile(A):
				if sys.platform!=E and not os.access(A,os.X_OK):return['sh',A]
				return[A]
			D=shutil.which(MAVEN_CMD)
			if D:return[D]
			raise FileNotFoundError(f"Maven not found. Neither '{MAVEN_CMD}' on PATH nor wrapper '{C}' at {B}.")
		E=P(G)+['-f',O,'-B','-ntp','-DfailIfNoTests=false',f"-Dtest={C}",'test']
		try:B=subprocess.Popen(E,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=_C,universal_newlines=_C)
		except FileNotFoundError as L:return TestExecutionFeedback(command=K.join(E),returncode=-1,test_class_name=C,timed_out=_B,passed=_B,tests_run=0,failures=0,errors=0,skipped=0,time_elapsed_sec=.0,issues=[],stdout_summary=str(L),stderr=str(L),reports_dirs=[],failure_reason_code=_D,failure_reason='Maven command not found')
		try:D,H=B.communicate(timeout=timeout)
		except subprocess.TimeoutExpired:B.kill();D,H=B.communicate();return TestExecutionFeedback(command=K.join(E),returncode=-1,test_class_name=C,timed_out=_C,passed=_B,tests_run=0,failures=0,errors=0,skipped=0,time_elapsed_sec=.0,issues=[],stdout_summary=_minimal_console_extract(D),stderr='Test execution timed out.',reports_dirs=[],failure_reason_code='TIMEOUT',failure_reason='Test execution timed out')
		M=_find_surefire_reports(G);A=_parse_surefire_reports(M)
		if A.tests_run==0 and(A.failures==0 and A.errors==0):
			N=_parse_console_summary(D)
			if N:A=N
		F=B.returncode==0 and A.tests_run>=0 and A.failures==0 and A.errors==0;I=_A;J=_A
		if not F:
			if A.failures or A.errors:I='TEST_ERRORS'if A.errors else'TEST_FAILURES';J=f"{A.failures} failure(s), {A.errors} error(s) across {A.tests_run} test(s)"
			elif B.returncode!=0:I,J=_detect_special_failure(D,H)
		Q=TestExecutionFeedback(command=K.join(E),returncode=B.returncode,test_class_name=C,timed_out=_B,passed=F,tests_run=A.tests_run,failures=A.failures,errors=A.errors,skipped=A.skipped,time_elapsed_sec=A.time_elapsed_sec,issues=A.issues,stdout_summary=_A if F else _minimal_console_extract(D),stderr=H or _A if not F else _A,reports_dirs=M,failure_reason_code=I,failure_reason=J);return Q
class _Agg:
	'Internal aggregation structure.'
	def __init__(A):A.tests_run=0;A.failures=0;A.errors=0;A.skipped=0;A.time_elapsed_sec=.0;(A.issues):List[TestIssue]=[]
def _find_surefire_reports(project_root:str)->List[str]:
	'Find all target/surefire-reports directories (handles multi-module builds).';C='surefire-reports';A:List[str]=[]
	for(B,D,E)in os.walk(project_root):
		if C in D and os.path.basename(B)=='target':A.append(os.path.join(B,C))
	return A
def _parse_surefire_reports(reports_dirs:List[str])->_Agg:
	M='error';L='time';K='testsuite';G='failure';A=_Agg()
	for H in reports_dirs:
		try:N=[A for A in os.listdir(H)if A.startswith('TEST-')and A.endswith('.xml')]
		except FileNotFoundError:continue
		for O in N:
			P=os.path.join(H,O)
			try:Q=ET.parse(P);C=Q.getroot()
			except Exception:continue
			E=[]
			if C.tag==K:E=[C]
			elif C.tag=='testsuites':E=list(C.findall(K))
			for B in E:
				A.tests_run+=_as_int(B.get('tests'));A.failures+=_as_int(B.get('failures'));A.errors+=_as_int(B.get('errors'));A.skipped+=_as_int(B.get('skipped'));A.time_elapsed_sec+=_as_float(B.get(L))
				for D in B.findall('testcase'):
					R=D.get('classname')or'';S=D.get('name')or'';T=_as_float(D.get(L))
					for I in(G,M):
						for F in D.findall(I):J=TestIssue(class_name=R,test_name=S,kind=G if I==G else M,message=F.get('message'),error_type=F.get('type'),time_sec=T,stack_trace=_truncate_stack_trace((F.text or'').strip()or _A));_maybe_fill_file_line(J);A.issues.append(J)
	return A
def _maybe_fill_file_line(issue:TestIssue)->_A:
	'Attempt to pull file and line from a Java stack trace fragment.';A=issue
	if not A.stack_trace:return
	B=re.search('\\(([^()/:\\\\\\s]+\\.java):(\\d+)\\)',A.stack_trace)
	if not B:B=re.search('([A-Za-z0-9_\\-./\\\\]+\\.java):(\\d+)',A.stack_trace)
	if B:
		A.file=B.group(1)
		try:A.line=int(B.group(2))
		except ValueError:pass
_SUMMARY_RE=re.compile('Tests run:\\s*(\\d+),\\s*Failures:\\s*(\\d+),\\s*Errors:\\s*(\\d+),\\s*Skipped:\\s*(\\d+)',re.IGNORECASE)
def _parse_console_summary(stdout:str)->Optional[_Agg]:
	"\n    Fallback when XML is missing: parse the last 'Tests run: ..., Failures: ...' summary\n    from Maven/Surefire console output.\n    ";D=stdout;E=list(_SUMMARY_RE.finditer(D or''))
	if not E:return
	B=E[-1];A=_Agg();A.tests_run=int(B.group(1));A.failures=int(B.group(2));A.errors=int(B.group(3));A.skipped=int(B.group(4));H=re.compile('Time elapsed:\\s*([0-9]*\\.?[0-9]+)\\s*s',re.IGNORECASE);F=(D or'').splitlines();C=_A
	for(I,J)in enumerate(F):
		if B.group(0)in J:C=I
	if C is not _A:
		K='\n'.join(F[max(0,C-20):C+20]);G=H.search(K)
		if G:
			try:A.time_elapsed_sec=float(G.group(1))
			except ValueError:pass
	return A
def _minimal_console_extract(stdout:Optional[str])->Optional[str]:
	"\n    Return the smallest useful slice from stdout:\n      * Last 'Tests run: ...' summary\n      * 'Failures:' / 'Errors:' listing lines\n      * 'There are test failures.' / 'BUILD FAILURE'\n      * 'No tests found...' / 'No tests were executed!'\n      * surefire plugin error lines\n    ";F=stdout
	if not F:return
	A=(F or'').splitlines();B=_A
	for G in range(len(A)-1,-1,-1):
		if'Results:'in A[G]:B=G;break
	C:List[str]=[]
	def H(line:str)->bool:
		F='No tests were executed';E='No tests found';D='There are test failures.';C='Errors:';B='Failures:';A=line
		if _SUMMARY_RE.search(A):return _C
		if A.startswith('[ERROR]'):G=B,C,D,_E,E,F,'Failed to execute goal',_F;return any(B in A for B in G)
		H=B,C,D,E,F;return any(A.strip().startswith(f"[INFO] {B}")for B in H)
	if B is not _A:J=A[max(0,B-15):min(len(A),B+80)];C=[A for A in J if H(A)]
	if not C:K=A[-200:];C=[A for A in K if H(A)]
	I=set();D=[]
	for E in C:
		if E not in I:D.append(E);I.add(E)
	return'\n'.join(D)if D else _A
def _detect_special_failure(stdout:Optional[str],stderr:Optional[str])->(str,str):
	A='SUREFIRE_PLUGIN_ERROR';B=f"{stdout or''}\n{stderr or''}";C=[('No tests (?:found|were executed)[:!]','NO_TESTS_MATCHED','No tests matched the -Dtest filter'),('Failed to execute goal .* surefire.*:',A,'Maven Surefire plugin failed'),(_F,A,'Surefire forked JVM failed'),(_E,_D,'Build failed')]
	for(D,E,F)in C:
		if re.search(D,B,re.IGNORECASE):return E,F
	return _D,'Build failed (see minimal console extract)'
def _truncate_stack_trace(text:Optional[str])->Optional[str]:
	A=text
	if not A:return
	B=A.splitlines()
	if len(B)<=MAX_STACKTRACE_LINES_ON_FAILURE:return A
	return'\n'.join(B[:MAX_STACKTRACE_LINES_ON_FAILURE])+'\n... (truncated)'
def _as_int(val:Optional[str])->int:
	try:return int(val)if val is not _A else 0
	except ValueError:return 0
def _as_float(val:Optional[str])->float:
	try:return float(val)if val is not _A else .0
	except ValueError:return .0
if __name__=='__main__':project_root='tests/resources/spring-petclinic/';execution_feedback=JavaExecution.execute(project_root,test_class_name='org.springframework.samples.petclinic.owner.OwnerControllerTests');print();print();print(execution_feedback)