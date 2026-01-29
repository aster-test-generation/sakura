import json
from enum import Enum
from typing import List,Any
def pretty_print(header:str,body:Any):
	D='=';A=header;A=A.upper();E=D*20+f" {A} "+D*20;F=D*len(E)
	def B(obj:Any)->Any:
		A=obj
		if hasattr(A,'model_dump'):A=A.model_dump(mode='json')
		if isinstance(A,Enum):return A.value
		if isinstance(A,dict):return{A:B(C)for(A,C)in A.items()}
		if isinstance(A,(list,tuple,set)):return[B(A)for A in A]
		return A
	C=B(body);print();print(E)
	if isinstance(C,(dict,list)):print(json.dumps(C,indent=4))
	else:print(C)
	print(F)