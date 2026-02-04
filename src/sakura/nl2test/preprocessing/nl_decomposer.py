from typing import List,Union
from sakura.nl2test.models.decomposition import GrammaticalBlock,GrammaticalBlockList,Scenario,DecompositionMode
from sakura.nl2test.preprocessing.decomposers import GrammaticalDecomposer,GherkinDecomposer,BaseDecomposer
from sakura.utils.llm import UsageTracker
class NLDecomposer:
	def __init__(A,mode:DecompositionMode=DecompositionMode.GRAMMATICAL,usage_tracker:UsageTracker|None=None):
		A.mode=mode;A.usage_tracker=usage_tracker or UsageTracker();B:BaseDecomposer
		if mode==DecompositionMode.GHERKIN:B=GherkinDecomposer(usage_tracker=A.usage_tracker)
		else:B=GrammaticalDecomposer(usage_tracker=A.usage_tracker)
		A._impl=B
	def decompose(A,nl_description:str)->Union[GrammaticalBlockList,Scenario]:return A._impl.decompose(nl_description)