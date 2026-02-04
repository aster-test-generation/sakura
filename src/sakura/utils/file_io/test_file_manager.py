from __future__ import annotations
_D='TestFileInfo'
_C=False
_B=None
_A=True
import os,re,tempfile,time
from pathlib import Path
from typing import TYPE_CHECKING,Annotated,List,Optional,Tuple,Union
from pydantic import BaseModel
from sakura.utils.exceptions.tool_exceptions import FileDeletionError
from sakura.utils.pretty.color_logger import RichLog
if TYPE_CHECKING:from sakura.nl2test.models import NL2TestInput;from sakura.test2nl.model.models import RoundTripTest
class TestFileInfo(BaseModel):
	'\n    Container for single generated test file.\n    ';qualified_class_name:Annotated[str,'The qualified class name of the test'];test_code:Annotated[str,'The test code of the test']='';id:Annotated[int,'The ID from NL2TestInput']=-1
	@classmethod
	def from_nl2test_input(B,nl2test_input:NL2TestInput,*,test_code:str='')->_D:A=nl2test_input;return B(qualified_class_name=A.qualified_class_name,test_code=test_code,id=A.id)
	@classmethod
	def from_roundtrip_test(B,rt_test:RoundTripTest)->_D:A=rt_test;return B(qualified_class_name=A.qualified_class_name,test_code=A.generated_test,id=-1)
class TestFileManager:
	def __init__(B,project_root:Path,test_base_dir:Union[str,Path]='src/test/java'):A=project_root;A.mkdir(parents=_A,exist_ok=_A);B.project_root=A;B.test_base_dir=A/Path(test_base_dir)
	@staticmethod
	def _atomic_write(target_path:Path,content:str)->_B:
		A=target_path;A.parent.mkdir(parents=_A,exist_ok=_A);B:Optional[str]=_B;C:Optional[int]=_B
		try:
			C,B=tempfile.mkstemp(dir=str(A.parent),prefix=f".{A.name}.",suffix='.tmp')
			with os.fdopen(C,'w',encoding='utf-8')as E:E.write(content);E.flush();os.fsync(E.fileno());C=_B
			os.replace(B,A);D:Optional[int]=_B
			if hasattr(os,'O_DIRECTORY'):
				try:D=os.open(str(A.parent),os.O_DIRECTORY)
				except OSError:D=_B
			if D is not _B:
				try:os.fsync(D)
				finally:os.close(D)
		finally:
			if C is not _B:
				try:os.close(C)
				except Exception:pass
			if B and os.path.exists(B):
				try:os.unlink(B)
				except Exception:pass
	@staticmethod
	def _sanitize_generated_code(code:str)->str:
		B=code
		if not isinstance(B,str):return B
		A=B.strip('\ufeff\r\n\t ');A=re.sub('^\\s*(?:```[^\\n]*\\n|(?:\'\'\'|\\"\\"\\")[\\t ]*\\n?)','',A);A=re.sub('(?:\\n?```|\\n?(?:\'\'\'|\\"\\"\\"))\\s*$','',A);return A
	@staticmethod
	def _split_qualified_class_name(qualified_class_name:str)->Tuple[str,str]:
		A=qualified_class_name;B,C,D=A.rpartition('.')
		if not C:return'',A
		return B,D
	@staticmethod
	def _package_dir_from_package(package:str)->Path:A=package;return Path(*A.split('.'))if A else Path()
	@staticmethod
	def _package_dir_from_qualified(qualified_class_name:str)->Path:A,B=TestFileManager._split_qualified_class_name(qualified_class_name);return TestFileManager._package_dir_from_package(A)
	@staticmethod
	def encode_class_name(id:int=-1)->str:
		'\n        Produce an encoded Java class name for a generated test case.\n        E.g., NL2T_001 for ID-based encoding\n        '
		if id==-1:raise ValueError('ID-based encoding requires a valid ID (id != -1)')
		A=f"NL2T_{id:03d}";return A
	@staticmethod
	def decode_class_name(encoded_class_name:str)->int:
		'\n        Decode the class name to an ID.\n        ';A=encoded_class_name
		if A.startswith('NL2T_'):A=A[5:]
		if not A.isdigit():raise ValueError(f"Invalid encoded class name format: expected numeric ID, got '{A}'. Legacy method signature encoding is no longer supported.")
		return int(A)
	@staticmethod
	def decode_file_name(file_name:str)->int:
		A=file_name
		if A.endswith('.java'):A=A[:-5]
		B=A.split('/')[-1];return TestFileManager.decode_class_name(B)
	def target_path(A,test_info:TestFileInfo,*,encode_class_name:bool=_A)->Path:
		'\n        Compute the target file path for a test file.\n        - If encode_class_name is True, use the encoded class name (e.g., NL2T_001.java)\n          under the package path derived from qualified_class_name.\n        - If encode_class_name is False, place the file directly under\n          `self.test_base_dir` mirroring the fully qualified class name\n          as a path, with `.java` appended at the end.\n        ';B=test_info
		if encode_class_name:C=A.encode_class_name(B.id);return A.test_base_dir/A._package_dir_from_qualified(B.qualified_class_name)/f"{C}.java"
		else:D=Path(*B.qualified_class_name.split('.'));return A.test_base_dir/D.with_suffix('.java')
	def _rewrite_java_header(F,package:str,new_class_name:str,code:str)->str:
		'\n        Ensure the Java package declaration and top-level type name match the target.\n\n        - Adds or replaces the package declaration with `package` (if non-empty),\n          or removes it if package is empty.\n        - Rewrites the first top-level class/interface/enum/record name to `new_class_name`.\n        ';B=package;A=code
		if B:
			C=f"package {B};";D='(?m)^\\s*package\\s+[A-Za-z_]\\w*(?:\\.[A-Za-z_]\\w*)*\\s*;'
			if re.search(D,A):A=re.sub(D,C,A,count=1)
			else:A=C+'\n\n'+A.lstrip()
		else:A=re.sub('(?m)^\\s*package\\s+[A-Za-z_]\\w*(?:\\.[A-Za-z_]\\w*)*\\s*;\\s*\\n?','',A,count=1)
		E='(?m)^(?P<prefix>\\s*(?:@\\w+(?:\\([^)]*\\))?\\s*)*(?:public\\s+)?(?:abstract\\s+|final\\s+)?(?:class|interface|enum|record)\\s+)(?P<name>[A-Za-z_]\\w*)';A=re.sub(E,f"\\g<prefix>{new_class_name}",A,count=1);return A
	def save_single(A,test_info:TestFileInfo,*,sync_names:bool=_C,encode_class_name:bool=_C,sanitize_wrappers:bool=_A,allow_overwrite:bool=_C)->Tuple[str,Path]:
		'\n        Save a single test file. If a conflict occurs, append a numeric suffix\n        to the class name (starting at 1) until a free filename is found unless\n        `allow_overwrite` is True, in which case the existing file is atomically\n        replaced.\n        ';C=test_info
		if encode_class_name:E=A.encode_class_name(C.id);D,L=A._split_qualified_class_name(C.qualified_class_name)
		else:D,E=A._split_qualified_class_name(C.qualified_class_name)
		G=A.test_base_dir/A._package_dir_from_package(D);G.mkdir(parents=_A,exist_ok=_A);B=E;F=G/f"{B}.java"
		if not allow_overwrite:
			I=1
			while F.exists():B=f"{E}{I}";F=G/f"{B}.java";I+=1
		H=A._sanitize_generated_code(C.test_code)if sanitize_wrappers else C.test_code;J=B!=E
		if sync_names or J:H=A._rewrite_java_header(D,B,H)
		A._atomic_write(F,H);K=f"{D}.{B}"if D else B;return K,F
	def save_batch(B,tests:List[TestFileInfo])->List[Path]:
		A:List[Path]=[]
		for C in tests:E,D=B.save_single(C,encode_class_name=_A);A.append(D)
		return A
	def load(B,test_info:TestFileInfo,*,encode_class_name:bool=_C)->str:
		A=B.target_path(test_info,encode_class_name=encode_class_name)
		if not A.exists():raise FileNotFoundError(f"Test file not found at {A}")
		with open(A,'r',encoding='utf-8')as C:return C.read()
	def make_test_fqn(A,test_info:TestFileInfo)->str:B=test_info;C=A.encode_class_name(B.id);D,E=A._split_qualified_class_name(B.qualified_class_name);return f"{D}.{C}"if D else C
	def delete_single(G,test_info:TestFileInfo,*,encode_class_name:bool=_C,strict:bool=_C,max_attempts:int=2,retry_delay:float=.05)->bool:
		'\n        Delete the test file at the location where it would have been saved.\n\n        - If `encode_class_name` is True, uses the encoded class name path.\n        - If `encode_class_name` is False, deletes the file at the fully\n          qualified class name path.\n\n        Returns True if the file existed and was successfully deleted, False otherwise.\n        When `strict` is True, an exception is raised if the file cannot be removed\n        after the configured retry attempts.\n        ';E=max_attempts;D=strict;A=G.target_path(test_info,encode_class_name=encode_class_name)
		if not A.exists():RichLog.info(f"File does not exist, skipping delete: {A}");return _A if D else _C
		B=0;C:Optional[Exception]=_B
		while B<E:
			B+=1
			try:A.unlink()
			except FileNotFoundError:break
			except Exception as F:
				C=F;RichLog.error(f"Failed to delete {A}: {F}")
				if B>=E:break
				time.sleep(max(retry_delay,.0))
			else:break
		if A.exists():
			if D:raise FileDeletionError(f"Failed to delete test file at {A}",extra_info={'path':str(A),'attempts':B,'error':str(C)if C else''})
			return _C
		return _A