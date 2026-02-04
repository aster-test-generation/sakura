from typing import List,Tuple
from langchain_core.tools import BaseTool
from.base import BaseCompositionTools
from.grammatical import GrammaticalCompositionTools
from.gherkin import GherkinCompositionTools
from sakura.nl2test.models import DecompositionMode
class CompositionTools:
	'Deprecated: use GrammaticalCompositionTools or GherkinCompositionTools.\n\n    Provided to keep legacy imports working.\n    '
	def __init__(A,*,decomposition_mode:DecompositionMode=DecompositionMode.GRAMMATICAL,**B):
		if decomposition_mode==DecompositionMode.GHERKIN:A._delegate=GherkinCompositionTools(**B)
		else:A._delegate=GrammaticalCompositionTools(**B)
	def all(A)->Tuple[List[BaseTool],List[BaseTool]]:return A._delegate.all()
	def __getattr__(A,item):return getattr(A._delegate,item)
__all__=['BaseCompositionTools','GrammaticalCompositionTools','GherkinCompositionTools','CompositionTools']