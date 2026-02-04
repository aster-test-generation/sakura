_J='^L\\d+'
_I='source'
_H='src/main/java'
_G='target'
_F='.java'
_E='utf-8'
_D=False
_C='.'
_B=True
_A=None
import glob,os,re,shutil,subprocess
from pathlib import Path
from typing import List,Tuple,Union,Dict
import ray
from bs4 import BeautifulSoup
from tqdm import tqdm
from sakura.utils.analysis import CommonAnalysis
from sakura.utils.pretty.color_logger import RichLog
from reaster.coverage.jacoco_test_watcher import TEST_CODE
from javabuild.build_factory import BuildFactory
from sakura.utils import constants
JACOCO_CLI_JAR=Path(__file__).parent.parent.parent.parent.parent.joinpath('resources/lib/jacococli-0.8.13.jar').absolute()
TEST_WATCHER_CLASS_NAME='JaCoCoTestWatcher'
JACOCO_TEST_FOLDER='target/jacoco-tests'
@ray.remote
def _collect_coverage_task(project_root:Path,source_root:str,test:Tuple[str,str]):
	D=project_root;A=test;B=A[0].replace('$','\\$');C=A[1].split('(')[0].strip();E=f"java -jar {JACOCO_CLI_JAR} report {JACOCO_TEST_FOLDER}{os.sep}{B}__{C}.exec --classfiles target/classes --sourcefiles {source_root} --html target{os.sep}{B}__{C}__report"
	try:RichLog.debug(f"Running command: {E}");H=subprocess.run(E,cwd=D,shell=_B,check=_B,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);G=IndividualTestCoverage.extract_all_covered_lines(D.joinpath(_G,f"{B}__{C}__report"));return A,G,_B
	except subprocess.CalledProcessError as F:RichLog.error(f'Error running command "{F.cmd}"');return A,F.stderr,_D
class IndividualTestCoverage:
	def __init__(A,project_root:Path,target_module:str=_A,build_type:str='maven',source_root:str=_H,test_root:str='src/test/java',junit_version:int=5):D=source_root;C=target_module;B=project_root;A.project_root=B;A.target_module=C;A.builder=BuildFactory.create(build_type,B,target_module=C);A.test_root=test_root;A.app_source_path=Path.cwd().joinpath(A.project_root,D).__str__();A.junit_version=junit_version;A.package_root=A.__get_root_package();A.source_root=D
	@classmethod
	def from_common_analysis(F,common:CommonAnalysis,*,project_root:Path,qualified_class_names:List[str],module_root:Path|_A=_A,build_type:str='maven',source_root:str=_H,junit_version:int=5)->'IndividualTestCoverage':
		D=module_root;C=common;B=project_root;A=Path(D)if D is not _A else _A
		if A is _A:
			for G in qualified_class_names:
				E=C.resolve_module_root(G)
				if E is not _A:A=E;break
		if A is not _A and not A.is_absolute():A=B/A
		H=A if A is not _A else B;I=C.resolve_test_base_dir(A,project_root=B);return F(project_root=H,build_type=build_type,source_root=source_root,test_root=str(I),junit_version=junit_version)
	def __get_root_package(A)->str:
		'\n        Computes and returns root package name for the application under test.\n\n        Computes root package by finding the common path prefix of all directories under application\n        source root that contain java source files.\n\n        Returns:\n            str: root package for app\n        ';B=[]
		for(C,E,I)in os.walk(A.app_source_path):
			for D in E:
				F=glob.glob(os.path.join(C,D,'*.java'),recursive=_D)
				if F:B.append(os.path.join(C,D))
		G=os.path.commonprefix(B);H=G.removeprefix(A.app_source_path).strip(os.path.sep).replace(os.path.sep,_C);return H
	def generate(A,tests_to_run:List[Tuple[str,str]]=[])->dict:
		'\n        Generate coverage for individual tests\n        Args:\n            tests_to_run: list of tests in (qualified_class_name, test_name) format\n\n        Returns:\n            dict[str, dict[str, dict]]\n            key: Test class name\n            value: key: test method name, value: coverage\n        ';B=tests_to_run;D=A.project_root.joinpath('pom_cov.xml');A.builder.add_code_coverage_dependencies(output_build_file=str(D),add_java_agent=_B);C=[]
		for E in B:C.append(E[0])
		C=list(set(C));F=IndividualTestCoverage.__get_test_class_info(test_root=str(A.project_root.joinpath(A.test_root)),qualified_names=C);G,H=A.__add_helper_class_modify_tests(all_test_classes=F)
		try:A.__run_tests(tests_to_run=B,modified_build_file=D,is_run_all_test=_B if len(B)==0 else _D);I,J=A.__collect_coverage(executed_tests=B if len(B)else A.__get_executed_test_methods());RichLog.info(f"Coverage for each test method computed; jacoco report failures on files: {J}")
		finally:A.__cleanup(original_class_content=G,additional_class=H)
		return I
	def __collect_coverage(D,executed_tests:List[Tuple[str,str]])->Tuple[Dict,List]:
		K='coverage_details';J='test_name';I='test_class_name';C={};F=[]
		for A in executed_tests:
			E=A[0];B=A[1].split('(')[0];G=f"java -jar {JACOCO_CLI_JAR} report {JACOCO_TEST_FOLDER}{os.sep}{E}__{B}.exec --classfiles target/classes --sourcefiles {D.source_root} --html target{os.sep}{E}__{B}__report"
			try:
				RichLog.info(f"Running command: {G}");M=subprocess.run(G,cwd=D.project_root,shell=_B,check=_B,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);H=D.extract_all_covered_lines(D.project_root.joinpath(_G,f"{E}__{B}__report"))
				if A[0]in C:C[A[0]].append({I:A[0],J:B,K:H})
				else:C[A[0]]=[{I:A[0],J:B,K:H}]
			except subprocess.CalledProcessError as L:RichLog.error(f'Error running command "{L.cmd}"');F.append(f"{E}__{B}.exec")
		RichLog.info(f"Coverage collected for {len(C.keys())} tests");return C,F
	@staticmethod
	def extract_all_covered_lines(jacoco_report_dir):
		B={};E=os.path.join(jacoco_report_dir,'**','*.html');F=glob.glob(E,recursive=_B)
		for A in F:
			if'index.html'not in A and'jacoco-sessions'not in A:
				C,D=IndividualTestCoverage.__extract_covered_lines_from_html(A)
				if C and D:B[C]=D
		return B
	@staticmethod
	def __extract_branch_lines_from_html(soup):
		C=soup.find('pre',class_=_I)
		if not C:return[],[]
		H=C.find_all('span',id=re.compile(_J));D=[];E=[];F=[]
		for G in H:
			A=int(G['id'][1:]);B=G.get('class',[])
			if'bfc'in B:D.append(A)
			elif'bnc'in B:E.append(A)
			elif'bpc'in B:F.append(A)
		return D,E,F
	@staticmethod
	def __extract_covered_lines_from_html(html_path):
		O='.html';N='span.el_source';M='a.el_package';L='html.parser';F='.java.html';C=html_path
		if F in C:
			with open(C,'r',encoding=_E)as D:A=BeautifulSoup(D,L)
			B=A.select_one('.breadcrumb')
			if not B:return _A,_A
			P=B.select_one(M)
			if not P:return _A,_A
			Q=B.select_one(N)
			if not Q:return _A,_A
			R=B.select_one(M).text.strip();S=B.select_one(N).text.strip().replace(_F,'');T=R+_C+S;G=A.find('pre',class_=_I);E=[]
			if G:
				for H in G.find_all('span',id=re.compile(_J)):
					U=int(H['id'][1:]);I=H.get('class',[])
					if'fc'in I or'pc'in I:E.append(U)
			V,W,X=IndividualTestCoverage.__extract_branch_lines_from_html(A)
			if Path(C.replace(F,O)).exists():
				with open(C.replace(F,O),'r',encoding=_E)as D:A=BeautifulSoup(D,L)
				J,K=IndividualTestCoverage.__extract_methods_with_coverage(A)
			else:J=K=0
			if not E:return _A,_A
			return T,{'covered_lines':E,'method_covered':J,'method_missed':K,'branch_lines_covered':V,'branch_lines_partial':X,'branch_lines_missed':W}
		return _A,_A
	@staticmethod
	def __extract_methods_with_coverage(soup:BeautifulSoup)->Tuple[list,list]:
		'\n        Extract covered and uncovered methods\n        Args:\n            soup:\n\n        Returns:\n\n        ';D=soup.find('table',class_='coverage')
		if not D:return[],[]
		E=[];A=[];G=D.find_all('tr')[1:]
		for H in G:
			B=H.find_all('td')
			if len(B)<3:continue
			I=B[0];J=B[2];F=I.find('a',class_='el_method')
			if not F:continue
			C=F.get_text(strip=_B);K=J.get_text(strip=_B).replace('%','')
			try:
				L=float(K)
				if L>0:E.append(C)
				else:A.append(C)
			except ValueError:A.append(C)
		return E,A
	def __run_tests(E,tests_to_run:list,modified_build_file:Path,is_run_all_test:bool=_B,clean_jacoco_test_dir:bool=_B):
		if clean_jacoco_test_dir:shutil.rmtree(E.project_root.joinpath(JACOCO_TEST_FOLDER),ignore_errors=_B)
		A=''
		if not is_run_all_test:
			B={}
			for D in tests_to_run:
				if D[0]not in B:B[D[0]]=[D[1]]
				else:B[D[0]].append(D[1])
			for C in B:
				if len(B[C])==1:A+=C+'#'+B[C][0].split('(')[0]+','
				else:
					A+=C+'#'+B[C][0].split('(')[0]
					for F in range(1,len(B[C])):A+='+'+B[C][F].split('(')[0]
					A+=','
			if A.endswith(','):A=A[:-1]
		E.builder.run_tests(target_tests=A if A else _A,build_file=str(modified_build_file))
	@staticmethod
	def __cleanup(original_class_content:dict,additional_class:Path)->_A:
		B=original_class_content;A=additional_class
		for C in B:
			with open(C,'w')as D:D.write(B[C])
		try:os.remove(A);RichLog.info(f"Helper file '{A}' deleted successfully.")
		except FileNotFoundError:RichLog.warn(f"File '{A}' does not exist.")
	def __get_executed_test_methods(E)->List[Tuple[str,str]]:
		D='.exec';A=[];B=E.project_root.joinpath(JACOCO_TEST_FOLDER)
		if not B.exists():return A
		F:List[Path]=[A for A in B.iterdir()if A.is_file()and A.name.endswith(D)]
		for G in F:C=G.name.removesuffix(D).split('__');A.append((C[0],'__'.join(C[1:])))
		return A
	def __add_helper_class_modify_tests(A,all_test_classes:List[Tuple[str,str]])->tuple[dict[str,str],Path]:
		'\n        Adds helper class and modified tests\n        Args:\n            all_test_classes: List of all qualified test names and their paths\n\n        Returns:\n            Tuple[dict, str]: original content, and the helper class\n\n        ';F={};RichLog.info('Modifying helper code for individual test');os.makedirs(os.path.dirname(A.project_root.joinpath(A.test_root)),exist_ok=_B)
		with open(A.project_root.joinpath(A.test_root,A.package_root.replace(_C,os.sep),f"{TEST_WATCHER_CLASS_NAME}.java"),'w')as C:G=TEST_CODE.replace('<package_name>',A.package_root);C.write(G)
		for E in all_test_classes:
			D=E[1]
			if not D:RichLog.warn(f"Skipping missing test class path for {E[0]}");continue
			with open(D,'r')as C:B=C.read()
			F[D]=B
			if _C.join(E[0].split(_C)[:-1])!=A.package_root:B=A.__add_import(B,f"import {A.package_root}.{TEST_WATCHER_CLASS_NAME};")
			if A.junit_version==5:B=A.__add_import(B,'import org.junit.jupiter.api.extension.ExtendWith;');B=A.__add_extends(B,f"@ExtendWith({TEST_WATCHER_CLASS_NAME}.class)")
			else:B=A.__add_import(B,'import org.junit.runner.RunWith;');B=A.__add_extends(B,f"@RunWith({TEST_WATCHER_CLASS_NAME}.class)")
			B=B.replace('@Disabled','//@Disabled')
			with open(D,'w')as C:C.write(B)
		return F,A.project_root.joinpath(A.test_root,A.package_root.replace(_C,os.sep),f"{TEST_WATCHER_CLASS_NAME}.java")
	@staticmethod
	def __add_import(code:str,import_statement:str)->str:
		'\n        Adds import statements to the test file\n        Args:\n            code:\n            import_statement:\n\n        Returns:\n\n        ';D='\n';C=import_statement;A=code
		if C in A:return A
		E=list(re.finditer('^import\\s+.*?;',A,re.MULTILINE))
		if E:G=E[-1];B=G.end();return A[:B]+D+C+A[B:]
		else:
			F=re.search('^package\\s+.*?;',A,re.MULTILINE)
			if F:B=F.end();return A[:B]+D+C+A[B:]
			else:return C+D+A
	@staticmethod
	def __add_extends(code:str,annotation:str)->str:
		'\n        Adds an annotation like @ExtendWith(...) or @RunWith(...) before the class declaration.\n        Special logic for @ExtendWith: it appends to existing list instead of duplicating.\n        ';B=annotation;A=code;H=B.startswith('@ExtendWith(');D=re.search('^(\\s*)(public\\s+)?(class|interface|enum)\\s+\\w+',A,re.MULTILINE)
		if not D:return A
		F=D.group(1);G=IndividualTestCoverage.__extract_class_from_annotation(B)
		if H:
			E=re.search('@ExtendWith\\s*\\(\\s*{?([^)}]*)}?[\\s]*\\)',A)
			if E:
				C=[A.strip()for A in E.group(1).split(',')]
				if G not in C:C.append(G);I=f"@ExtendWith({{{', '.join(C)}}})"if len(C)>1 else f"@ExtendWith({C[0]})";A=A[:E.start()]+I+A[E.end():]
				return A
			else:return IndividualTestCoverage.__insert_annotation(A,B,F,D.start())
		else:
			if B in A:return A
			return IndividualTestCoverage.__insert_annotation(A,B,F,D.start())
	@staticmethod
	def __insert_annotation(code:str,annotation:str,indent:str,position:int)->str:A=position;return code[:A]+f"{indent}{annotation}\n"+code[A:]
	@staticmethod
	def __extract_class_from_annotation(annotation:str)->str:
		"\n        From @ExtendWith(MyExtension.class), extract 'MyExtension.class'\n        ";A=annotation;B=re.match('@\\w+\\(\\s*{?([^)}]*)}?[\\s]*\\)',A)
		if B:return B.group(1).split(',')[0].strip()
		return A
	@staticmethod
	def __find_all_test_classes(test_root:str)->List[Tuple[str,str]]:
		'\n        Walk through all Java files under test_root and return a list of:\n        (fully_qualified_class_name, file_path)\n        ';A=[]
		for(G,M,H)in os.walk(test_root):
			for B in H:
				if B.endswith(_F):
					C=os.path.join(G,B)
					with open(C,'r',encoding=_E,errors='ignore')as I:D=I.read()
					E=re.search('^\\s*package\\s+([\\w.]+);',D,re.MULTILINE)
					if not E:continue
					J=E.group(1);F=re.search('\\b(public\\s+)?(class|interface|enum)\\s+(\\w+)',D)
					if not F:continue
					K=F.group(3);L=f"{J}.{K}";A.append((L,C))
		return A
	@staticmethod
	def __resolve_class_paths(test_root:str,qualified_names:List[str])->List[Tuple[str,str]]:
		'\n        Given a list of qualified names, return list of:\n        (qualified_name, file_path)\n        ';A=[]
		for B in qualified_names:
			C=os.path.join(test_root,*B.split(_C))+_F
			if os.path.isfile(C):A.append((B,C))
			else:A.append((B,_A))
		return A
	@staticmethod
	def __get_test_class_info(test_root:str,qualified_names:Union[_A,List[str]]=_A)->list[Tuple[str,str]]:
		'\n        Given a list of qualified test class name and their path\n        Args:\n            test_root:\n            qualified_names:\n\n        Returns:\n\n        ';B=qualified_names;A=test_root
		if len(B)==0:return IndividualTestCoverage.__find_all_test_classes(A)
		else:return IndividualTestCoverage.__resolve_class_paths(A,B)