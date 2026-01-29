'Statistical utilities for computing distribution summaries.'
from __future__ import annotations
_A=.0
import math
from dataclasses import dataclass
from typing import Any,Dict,List,Mapping
@dataclass(frozen=True)
class DistributionSummary:mean:float;std:float;min:float;p10:float;p25:float;p50:float;p75:float;p90:float;max:float;count:int;total:float
def percentile(sorted_values:List[float],percentile_value:float)->float:
	'Calculate percentile value from a sorted list.';A=sorted_values
	if not A:return _A
	if len(A)==1:return A[0]
	C=percentile_value/100*(len(A)-1);B=int(math.floor(C));D=int(math.ceil(C))
	if B==D:return A[B]
	E=A[B];F=A[D];G=C-B;return E+G*(F-E)
def summarize_distribution(values:List[float])->DistributionSummary:
	'Compute distribution summary statistics for a list of values.';A=values
	if not A:return DistributionSummary(mean=_A,std=_A,min=_A,p10=_A,p25=_A,p50=_A,p75=_A,p90=_A,max=_A,count=0,total=_A)
	B=sorted(A);C=sum(A);D=C/len(A);E=sum((A-D)**2 for A in A)/len(A);F=math.sqrt(E);return DistributionSummary(mean=D,std=F,min=B[0],p10=percentile(B,10),p25=percentile(B,25),p50=percentile(B,50),p75=percentile(B,75),p90=percentile(B,90),max=B[-1],count=len(A),total=C)
def distribution_to_dict(summary:DistributionSummary)->Dict[str,Any]:'Convert a DistributionSummary to a dictionary.';A=summary;return{'mean':A.mean,'std':A.std,'min':A.min,'p10':A.p10,'p25':A.p25,'p50':A.p50,'p75':A.p75,'p90':A.p90,'max':A.max,'count':A.count,'total':A.total}
def distributions_to_dict(distributions:Mapping[str,DistributionSummary])->Dict[str,Any]:'Convert a mapping of distribution summaries to a dictionary.';return{A:distribution_to_dict(B)for(A,B)in distributions.items()}
def build_distributions(metrics:Mapping[str,List[float]])->Dict[str,DistributionSummary]:'Build distribution summaries for each metric in the mapping.';return{A:summarize_distribution(B)for(A,B)in metrics.items()}