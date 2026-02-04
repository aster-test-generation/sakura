_D='[INFO]'
_C='[ERROR]'
_B=False
_A=None
import re
from pathlib import Path
from typing import List,Optional,Set,Tuple
from javabuild.maven_build import MavenBuild
from pydantic import BaseModel,Field
from sakura.utils.analysis import CommonAnalysis
ErrorKey=Tuple[str,int,Optional[int],str]
class CompilationError(BaseModel):file:str;line:int;column:Optional[int]=_A;message:str;details:List[str]=Field(default_factory=list,description='Compiler diagnostic details')
class CompilationScopeResult(BaseModel):success:bool;output:str;errors:List[CompilationError]
class JavaMavenCompilation(MavenBuild):
	"\n    Wrapper over javabuild's MavenBuild that adds structured compiler diagnostics.\n    ";DETAIL_SKIP_PREFIXES='-> [Help','For more information about the errors','Re-run Maven';ERROR_HEADER_RE=re.compile('^\\[ERROR\\]\\s+(?P<path>.*?\\.java):\\[(?P<line>\\d+)(?:,(?P<col>\\d+))?\\]\\s+(?P<msg>.*)$');ERROR_INDENT_RE=re.compile('^\\[ERROR\\]\\s{2,}\\S')
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
	def compile_scope(A,timeout:int|_A=600)->CompilationScopeResult:
		C=timeout
		if A.target_module:E=A.install_selected_projects_skip_tests_proc(timeout=C)
		B=A.compile_tests_proc(pre_compile_build=True,also_make=_B,timeout=C);D=A.parse_compilation_errors(B.stdout);return CompilationScopeResult(success=B.returncode==0,output=B.stdout,errors=D)
	def get_compilation_errors(A,timeout:int|_A=600)->List[CompilationError]:'\n        Run `mvn test-compile` via MavenBuild and parse every compiler diagnostic.\n        ';return A.compile_scope(timeout=timeout).errors
	def parse_compilation_errors(A,compiler_output:str)->List[CompilationError]:
		I=A._filter_lines(compiler_output);D:List[CompilationError]=[];F:Set[ErrorKey]=set();B:Optional[CompilationError]=_A
		for E in I:
			C=A.ERROR_HEADER_RE.match(E)
			if C:A._flush_current(B,D,F);J=C.group('path').strip();K=CommonAnalysis.normalize_path_in_project(J,A.project_root);G=C.group('col');B=CompilationError(file=K,line=int(C.group('line')),column=int(G)if G is not _A else _A,message=C.group('msg').strip());continue
			if B and A._is_detail_line(E):
				H=A._sanitize_detail(E)
				if not A._should_skip_detail(H):B.details.append(H)
		A._flush_current(B,D,F);return D
	@classmethod
	def _filter_lines(D,text:str)->List[str]:
		'Drop noisy lines that never contribute to compiler diagnostics.';B:List[str]=[]
		for C in text.splitlines():
			A=C.rstrip('\r\n')
			if A.startswith(_D)and'COMPILATION ERROR'not in A:continue
			if A.startswith('Downloading ')or A.startswith('Downloaded '):continue
			B.append(A)
		return B
	@classmethod
	def _is_detail_line(B,line:str)->bool:
		'True when the line adds context to the current error block.';A=line
		if B.ERROR_HEADER_RE.match(A):return _B
		if A.startswith(_D)or A.startswith('[WARNING]'):return _B
		if not A.strip():return _B
		if A.startswith(_C):return bool(B.ERROR_INDENT_RE.match(A))
		return A.startswith('  ')or A.startswith('\t')
	@classmethod
	def _sanitize_detail(B,line:str)->str:
		'Remove prefixes that Maven adds before the actual diagnostic text.';A=line
		if A.startswith(_C):A=A[len(_C):].lstrip()
		if A.strip()=='^':return'^'
		return A.lstrip()
	@classmethod
	def _should_skip_detail(A,detail:str)->bool:return any(detail.startswith(A)for A in A.DETAIL_SKIP_PREFIXES)
	@classmethod
	def _flush_current(D,current:Optional[CompilationError],errors:List[CompilationError],seen_keys:Set[ErrorKey])->_A:
		B=seen_keys;A=current
		if not A:return
		C:ErrorKey=(A.file,A.line,A.column,A.message)
		if C not in B:B.add(C);errors.append(A)