_G=False
_F='append'
_E='csv'
_D=True
_C='utf-8'
_B='json'
_A=None
import csv,json
from pathlib import Path
from typing import Any,List,Literal,Sequence,Type,TypeVar,Union
from pydantic import BaseModel
from sakura.utils.pretty.color_logger import RichLog
SubModel=TypeVar('SubModel',bound=BaseModel)
class StructuredDataManager:
	def __init__(A,base_dir:Path):A.base_dir=base_dir;A.base_dir.mkdir(parents=_D,exist_ok=_D)
	@staticmethod
	def _as_list_of_dicts(data:Union[Sequence[BaseModel],Sequence[dict],BaseModel,dict])->List[dict]:
		'Normalize input data to a list of dictionaries';A=data
		if A is _A:return[]
		def B(x:Any)->dict:
			if isinstance(x,BaseModel):return x.model_dump(mode=_B)
			if isinstance(x,dict):return x
			if hasattr(x,'__dict__'):return dict(x.__dict__)
			raise TypeError(f"Unsupported item type: {type(x)}")
		if isinstance(A,(list,tuple)):return[B(A)for A in A]
		return[B(A)]
	@staticmethod
	def _atomic_write_text(path:Path,content:str)->_A:
		'To avoid any writing corruption, we write to a temporary file then replace the original.';A=path;B=A.with_suffix(A.suffix+'.tmp')
		with B.open('w',encoding=_C,newline='')as C:C.write(content)
		B.replace(A)
	def save(D,file_name:str,data:Union[Sequence[BaseModel],Sequence[dict],BaseModel,dict],*,format:Literal[_B,_E]=_B,mode:Literal['write',_F]='write')->_A:
		A=D.base_dir/file_name;C=D._as_list_of_dicts(data)
		if format==_B:
			if mode==_F and A.exists():
				try:
					with A.open('r',encoding=_C)as G:B=json.load(G)
				except Exception:B=[]
				if not isinstance(B,list):B=[B]
				B.extend(C);D._atomic_write_text(A,json.dumps(B,indent=4,ensure_ascii=_D))
			else:D._atomic_write_text(A,json.dumps(C,indent=4,ensure_ascii=_D))
		elif format==_E:
			K=A.exists();E=mode==_F and K
			if not C:return
			F=_A
			if E:
				try:
					with A.open('r',encoding=_C,newline='')as L:I=csv.DictReader(L);F=list(I.fieldnames)if I.fieldnames else _A
				except Exception:F=_A
				if F is _A:E=_G
			H=F or sorted({B for A in C for B in A.keys()})
			with A.open('a'if E else'w',encoding=_C,newline='')as G:
				J=csv.DictWriter(G,fieldnames=H)
				if not E and H:J.writeheader()
				for M in C:J.writerow({A:M.get(A,'')for A in H})
		else:raise ValueError(f"Unsupported format: {format}")
	def load(E,file_name:str,model_cls:Type[SubModel],*,format:str=_B)->List[SubModel]:
		D=model_cls;B=E.base_dir/file_name
		if not B.exists():raise FileNotFoundError(f"File not found: {B}")
		if format==_B:
			with B.open('r',encoding=_C)as C:A=json.load(C)
			if isinstance(A,dict):A=[A]
			return[D(**A)for A in A]
		elif format==_E:
			with B.open('r',encoding=_C)as C:F=csv.DictReader(C);A=list(F)
			return[D(**A)for A in A]
		else:raise ValueError(f"Unsupported format: {format}")
	def delete(B,file_name:str)->bool:
		A=B.base_dir/file_name
		try:A.unlink();return _D
		except FileNotFoundError:return _G
		except Exception as C:RichLog.error(f"Failed to delete {A}: {C}");return _G
	def delete_many(A,file_names:Sequence[str])->_A:
		for B in file_names:A.delete(B)