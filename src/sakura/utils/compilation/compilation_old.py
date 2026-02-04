_G='message'
_F='column'
_E='[INFO]'
_D='line'
_C='[ERROR]'
_B='path'
_A=None
import os,sys,re,shutil,subprocess
from pathlib import Path
from typing import List,Tuple,Dict,Optional,Union
from pydantic import BaseModel,Field
from sakura.utils.constants import MAVEN_CMD
class CompilationError(BaseModel):file:str;line:int;column:Optional[int]=_A;message:str;details:List[str]=Field(default_factory=list)
class JavaCompilation:
	'\n    Compile and prepare Maven-based Java project for testing, and parse compiler errors.\n    '
	@staticmethod
	def _normalize_project_root(project_root:Union[str,Path])->Tuple[Path,Path]:
		'\n        Returns (abs_project_root, pom_path) and verifies pom.xml exists.\n        ';B=Path(project_root).expanduser().resolve();A=B/'pom.xml'
		if not A.is_file():raise FileNotFoundError(f"pom.xml not found at {A}")
		return B,A
	@staticmethod
	def _resolve_maven_cmd_parts(project_root:Union[str,Path])->List[str]:
		'\n        Resolves the correct Maven command: mvnw wrapper if present (with sh on *nix if needed),\n        otherwise MAVEN_CMD on PATH. Returns parts ready to prepend to args.\n        ';E='win32';B=project_root;C='mvnw.cmd'if sys.platform==E else'mvnw';F=Path(B);A=F/C
		if A.is_file():
			if sys.platform!=E and not os.access(A,os.X_OK):return['sh',str(A)]
			return[str(A)]
		D=shutil.which(MAVEN_CMD)
		if D:return[D]
		raise FileNotFoundError(f"Maven not found. Neither '{MAVEN_CMD}' on PATH nor wrapper '{C}' at {B}.")
	@staticmethod
	def _run_mvn(project_root:Union[str,Path],cmd_base:List[str],args:List[str])->subprocess.CompletedProcess:return subprocess.run(cmd_base+args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,cwd=str(Path(project_root)))
	@staticmethod
	def _run_compile(project_root:Union[str,Path],pom_path:Union[str,Path],cmd_base:List[str])->subprocess.CompletedProcess:return JavaCompilation._run_mvn(project_root,cmd_base,['-f',str(Path(pom_path).resolve()),'-B','-Dstyle.color=never','clean','test-compile'])
	@staticmethod
	def _should_try_formatting(stdout:str,stderr:str)->bool:A=f"{stdout or''}\n{stderr or''}".lower();B='spring-javaformat','spring-javaformat:check','spring javaformat','format violation',"please run 'mvn spring-javaformat:apply'";return any(B in A for B in B)
	@staticmethod
	def _apply_format(project_root:Union[str,Path],pom_path:Union[str,Path],cmd_base:List[str])->bool:A=JavaCompilation._run_mvn(project_root,cmd_base,['-f',str(Path(pom_path).resolve()),'spring-javaformat:apply']);return A.returncode==0
	@staticmethod
	def _compile_with_auto_format(project_root:Union[str,Path])->Tuple[subprocess.CompletedProcess,str]:
		'\n        Performs compilation (clean test-compile). If formatting is requested by plugins,\n        applies it and re-compiles. Returns (CompletedProcess, combined_output).\n        ';C,D=JavaCompilation._normalize_project_root(project_root);E=JavaCompilation._resolve_maven_cmd_parts(C);A=JavaCompilation._run_compile(C,D,E)
		if A.returncode!=0 and JavaCompilation._should_try_formatting(A.stdout,A.stderr):
			if JavaCompilation._apply_format(C,D,E):A=JavaCompilation._run_compile(C,D,E)
		B=f"{A.stdout or''}\n{A.stderr or''}"
		if len(B)>=2 and B[0]=="'"and B[-1]=="'":B=B[1:-1]
		return A,B
	@staticmethod
	def _report_path(filepath:str,project_root:Optional[Union[str,Path]])->str:
		'\n        Returns a modified path with no user prefix.\n        ';C=project_root;B=filepath
		if not B:return B
		A=B.replace('\\','/')
		if C:
			F=str(Path(C).expanduser().resolve()).replace('\\','/');D=F+'/'
			if A.lower().startswith(D.lower()):return A[len(D):]
		E=A.find('/src/')
		if E!=-1:return A[E+1:]
		return os.path.basename(A)
	@staticmethod
	def _iter_lines(text_or_lines:Union[str,List[str]])->List[str]:
		A=text_or_lines
		if isinstance(A,str):B=A.splitlines()
		else:B=list(A)
		return[A.rstrip('\r\n')for A in B]
	@staticmethod
	def _is_new_error_line(line:str)->bool:'\n        True if the line introduces a new compiler error for a .java source file.\n        ';return bool(re.match('^\\[ERROR\\]\\s+.*?\\.java:\\[\\d+(?:,\\d+)?\\]\\s+.*',line))
	@staticmethod
	def _is_detail_line(line:str)->bool:
		'\n        Lines that are part of the current error explanation. Maven sometimes prefixes them with [ERROR],\n        sometimes they are bare indented lines.\n        ';B=False;A=line
		if JavaCompilation._is_new_error_line(A):return B
		if A.startswith(_E)or A.startswith('[WARNING]'):return B
		C=A.lstrip()
		if not C:return B
		if A.startswith(_C):return bool(re.match('^\\[ERROR\\]\\s{2,}\\S',A))
		return A.startswith('  ')or A.startswith('\t')
	@staticmethod
	def _parse_error_header(line:str)->Optional[Dict[str,Union[str,int]]]:
		'\n        Parses an error header line like:\n        [ERROR] /path/to/Foo.java:[12,34] cannot find symbol\n        or\n        [ERROR] C:\\p\\Foo.java:[8,2] error: package x.y does not exist\n        Returns dict with file, line, column, message (raw).\n        ';A=re.match('^\\[ERROR\\]\\s+(?P<path>.*?\\.java):\\[(?P<line>\\d+)(?:,(?P<col>\\d+))?\\]\\s+(?P<msg>.*)$',line)
		if not A:return
		C=A.group(_B).strip();D=int(A.group(_D));B=A.group('col');E=int(B)if B is not _A else _A;F=A.group('msg').strip();return{_B:C,_D:D,_F:E,_G:F}
	@staticmethod
	def _sanitize_detail(line:str)->str:
		'\n        Removes [ERROR] prefix and leading spaces, keeps core content.\n        ';A=line
		if A.startswith(_C):A=A[len(_C):].lstrip()
		return A.lstrip()
	@staticmethod
	def _parse_errors_from_lines(lines:List[str],project_root:Optional[Union[str,Path]])->Tuple[List[str],List[CompilationError]]:
		'\n        Core parser. Walks the lines once, builds a set of erroneous files (.java), and\n        a structured list of compiler errors with important details.\n        ';H:set=set();G:List[CompilationError]=[];A:Optional[CompilationError]=_A;D:set=set()
		for J in lines:
			E=J.rstrip()
			if JavaCompilation._is_new_error_line(E):
				B=JavaCompilation._parse_error_header(E)
				if B:
					if A is not _A:
						C=A.file,A.line,A.column,A.message
						if C not in D:D.add(C);G.append(A)
					K=JavaCompilation._report_path(B[_B],project_root);I=os.path.basename(B[_B].replace('\\','/'))
					if I.endswith('.java'):H.add(I)
					A=CompilationError(file=K,line=B[_D],column=B[_F],message=B[_G])
				continue
			if A is not _A and JavaCompilation._is_detail_line(E):
				F=JavaCompilation._sanitize_detail(E)
				if F.startswith('-> [Help'):continue
				if F.startswith('For more information about the errors'):continue
				if F.startswith('Re-run Maven'):continue
				A.details.append(F);continue
		if A is not _A:
			C=A.file,A.line,A.column,A.message
			if C not in D:D.add(C);G.append(A)
		return sorted(H),G
	@staticmethod
	def parse_compilation_errors(text_or_lines:Union[str,List[str]],project_root:Optional[Union[str,Path]]=_A)->Tuple[List[str],List[CompilationError]]:
		"\n        Parse Maven compiler output (string or list of lines), remove noise like INFO lines,\n        and return (erroneous_files, errors) without running Maven. The first value is a sorted\n        list of .java filenames with compilation errors, not FQCNs.\n\n        Example:\n            (['MyTest.java'], [CompilationError(file='src/test/java/com/acme/MyTest.java', line=12, column=8, message='cannot find symbol', details=['symbol:   class Foo', 'location: class com.acme.MyTest'])])\n        ";C=JavaCompilation._iter_lines(text_or_lines);B:List[str]=[]
		for A in C:
			if A.startswith(_E):
				if'COMPILATION ERROR'not in A:continue
			if A.startswith('Downloading ')or A.startswith('Downloaded '):continue
			B.append(A)
		return JavaCompilation._parse_errors_from_lines(B,project_root)
	@staticmethod
	def get_erroneous_files_and_errors(project_root:Union[str,Path])->Tuple[List[str],List[CompilationError]]:"\n        Run Maven (clean test-compile), auto-apply Spring JavaFormat if requested,\n        and return both the erroneous files and the parsed compilation errors.\n\n        Returns a tuple (erroneous_files, errors) where erroneous_files is a sorted list of\n        Java source filenames with compilation errors (e.g., ['MyTest.java']).\n        ";A=project_root;C,B=JavaCompilation._compile_with_auto_format(A);return JavaCompilation.parse_compilation_errors(B,A)
	@staticmethod
	def get_erroneous_files(project_root:Union[str,Path])->List[str]:A,B=JavaCompilation.get_erroneous_files_and_errors(project_root);return A
if __name__=='__main__':
	project_root='tests/resources/spring-petclinic/';error_files,errors=JavaCompilation.get_erroneous_files_and_errors(project_root);print('Erroneous files:',error_files);print('Errors:')
	for(i,e)in enumerate(errors,1):
		col=f",{e.column}"if e.column is not _A else'';print(f"{i}. {e.file}:{e.line}{col} -> {e.message}")
		for d in e.details:print(f"   - {d}")