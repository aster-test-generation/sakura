import re
from typing import Optional
class FormatValidator:
	'Utility helpers for massaging formatter-related LLM responses.';_FENCE_LANGUAGE_HINTS={'java','javascript','js','typescript','ts','python','py','c','c++','cpp','csharp','c#','cs','go','golang','kotlin','swift','scala','groovy','ruby','php','bash','sh','shell','powershell','ps','ps1','sql','json','yaml','yml','xml','html','css','text','plain','plaintext','markdown','md'}
	@staticmethod
	def sanitize_code_block(code:str)->str:
		"\n        Strip markdown or quote wrappers like ```java ... ``` or ''' ... '''.\n        ";C=code
		if not isinstance(C,str):return C
		A=C.strip()
		if not A:return A
		for D in('```','~~~'):
			B=FormatValidator._strip_fenced_block(A,D)
			if B is not None:return B
		for E in("'''",'"""'):
			B=FormatValidator._strip_simple_wrapper(A,E)
			if B is not None:return B
		return A
	@staticmethod
	def _strip_fenced_block(text:str,fence:str)->Optional[str]:
		B=text;A=fence
		if not B.startswith(A):return
		D=B.rfind(A)
		if D<=len(A):return
		C=B[len(A):D];C=FormatValidator._strip_language_hint(C);return C.strip()
	@staticmethod
	def _strip_language_hint(inner:str)->str:
		A=inner.lstrip('\r')
		if A.startswith('\n'):return A.lstrip('\r\n')
		C=A.lstrip(' \t');B=re.match('([A-Za-z0-9_+\\-#.]+)([\\t ]+|\\r?\\n)',C)
		if B and FormatValidator._looks_like_language_hint(B.group(1)):return C[B.end():]
		return A
	@staticmethod
	def _looks_like_language_hint(token:str)->bool:
		A=token.strip().lower()
		if not A:return False
		A=A.replace('language-','');return A in FormatValidator._FENCE_LANGUAGE_HINTS
	@staticmethod
	def _strip_simple_wrapper(text:str,wrapper:str)->Optional[str]:
		B=text;A=wrapper
		if B.startswith(A)and B.endswith(A):C=B[len(A):-len(A)];C=FormatValidator._strip_language_hint(C);return C.strip()