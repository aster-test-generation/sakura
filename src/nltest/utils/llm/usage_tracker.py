from __future__ import annotations
from dataclasses import dataclass
@dataclass
class UsageTracker:
	calls:int=0;input_tokens:int=0;output_tokens:int=0
	def record(A,input_tokens:int,output_tokens:int)->None:'Accumulate token usage for a single LLM invocation.';A.calls+=1;A.input_tokens+=input_tokens or 0;A.output_tokens+=output_tokens or 0
	def reset(A)->None:'Clear all tracked counts.';A.calls=0;A.input_tokens=0;A.output_tokens=0
	def totals(A)->dict[str,int]:'Return a snapshot of the tracked counts.';return{'calls':A.calls,'input_tokens':A.input_tokens,'output_tokens':A.output_tokens}