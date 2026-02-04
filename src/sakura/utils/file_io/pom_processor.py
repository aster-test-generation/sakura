from __future__ import annotations
_A=None
from pathlib import Path
from typing import Dict,List,Sequence,Tuple
import re,xml.etree.ElementTree as ET
from pydantic import BaseModel
from sakura.utils.exceptions import PomXmlNotFoundError
class MavenDependency(BaseModel):'Represents a Maven dependency coordinate.';group_id:str;artifact_id:str
class PomProcessor:
	@staticmethod
	def identify_dependencies(module_root:Path,parent_roots:Sequence[Path]|_A=_A)->List[MavenDependency]:
		'\n        Identify Maven dependencies for a module, including inherited parent POMs.\n        Raises PomXmlNotFoundError if the module pom.xml is missing.\n        Returns an empty list if the module POM exists but cannot be parsed.\n        ';c='version';b='parent';a='project.version';Z='project.artifactId';Y='project.groupId';O='artifactId';N='groupId';K=module_root;J='}';I='pom.xml';C='';K=Path(K);d=[Path(A)for A in parent_roots or[]];H=K/I
		if not H.exists()or not H.is_file():raise PomXmlNotFoundError('Module pom.xml not found.',extra_info={'pom_path':str(H)})
		P:List[Tuple[Path,ET.Element,str]]=[];Q:set[Path]=set();e=re.compile('\\$\\{([^}]+)\\}')
		def f(pom_path:Path)->Tuple[ET.Element,str]|_A:
			try:A=ET.parse(pom_path).getroot()
			except ET.ParseError:return
			B=C
			if A.tag.startswith('{')and J in A.tag:B=A.tag[1:A.tag.find(J)]
			return A,B
		def G(namespace:str,tag:str)->str:A=namespace;return f"{{{A}}}{tag}"if A else tag
		def D(element:ET.Element|_A,tag:str,namespace:str)->str:
			B=element
			if B is _A:return C
			A=B.find(G(namespace,tag))
			if A is _A or A.text is _A:return C
			return A.text.strip()
		def g(root:ET.Element,namespace:str)->Dict[str,str]:
			B=root.find(G(namespace,'properties'))
			if B is _A:return{}
			D:Dict[str,str]={}
			for E in list(B):
				A=E.tag
				if A.startswith('{')and J in A:A=A[A.find(J)+1:]
				F=(E.text or C).strip()
				if A and F:D[A]=F
			return D
		def R(value:str,properties:Dict[str,str])->str:
			A=value
			if not A:return C
			def B(match:re.Match[str])->str:A=match;B=A.group(1);return properties.get(B,A.group(0))
			return e.sub(B,A)
		def h(properties:Dict[str,str],*,group_id:str,artifact_id:str,version:str,parent_group_id:str,parent_artifact_id:str,parent_version:str)->Dict[str,str]:
			G=parent_version;F=parent_artifact_id;E=parent_group_id;D=version;C=artifact_id;B=group_id;A=dict(properties)
			if B:A[Y]=B;A['pom.groupId']=B
			if C:A[Z]=C;A['pom.artifactId']=C
			if D:A[a]=D;A['pom.version']=D
			if E:A['project.parent.groupId']=E;A['parent.groupId']=E
			if F:A['project.parent.artifactId']=F;A['parent.artifactId']=F
			if G:A['project.parent.version']=G;A['parent.version']=G
			return A
		def i(root:ET.Element,namespace:str,properties:Dict[str,str])->List[MavenDependency]:
			B=properties;A=namespace;C=root.find(G(A,'dependencies'))
			if C is _A:return[]
			E:List[MavenDependency]=[]
			for F in C.findall(G(A,'dependency')):
				H=R(D(F,N,A),B);I=R(D(F,O,A),B)
				if H and I:E.append(MavenDependency(group_id=H,artifact_id=I))
			return E
		def j(root:ET.Element,namespace:str,base_dir:Path)->Path|_A:
			F=namespace;H=root.find(G(F,b))
			if H is _A:return
			J=H.find(G(F,'relativePath'))
			if J is _A:A=Path('..')/I
			else:
				K=(J.text or C).strip()
				if not K:A=_A
				else:A=Path(K)
			B:List[Path]=[]
			if A is not _A:B.append(base_dir/A)
			for D in d:
				if D.name==I:B.append(D)
				else:B.append(D/I)
			for E in B:
				if E.exists()and E.is_file():return E
		E=H
		while E is not _A and E not in Q:
			Q.add(E);S=f(E)
			if S is _A:
				if E==H:return[]
				break
			B,A=S;P.append((E,B,A));E=j(B,A,E.parent)
		T:List[MavenDependency]=[];U:set[tuple[str,str]]=set();F:Dict[str,str]={}
		for(o,B,A)in reversed(P):
			L=B.find(G(A,b));V=D(L,N,A);k=D(L,O,A);W=D(L,c,A);l=D(B,N,A)or V or F.get(Y,C);m=D(B,O,A)or F.get(Z,C);n=D(B,c,A)or W or F.get(a,C);F={**F,**g(B,A)};F=h(F,group_id=l,artifact_id=m,version=n,parent_group_id=V,parent_artifact_id=k,parent_version=W)
			for M in i(B,A,F):
				X=M.group_id,M.artifact_id
				if X not in U:T.append(M);U.add(X)
		return T