from typing import List,Tuple
from langchain_core.tools import BaseTool
from.base import BaseLocalizationTools
from.grammatical import GrammaticalLocalizationTools
from.gherkin import GherkinLocalizationTools
from sakura.nl2test.models import DecompositionMode
class LocalizationTools:
	'Deprecated: use GrammaticalLocalizationTools or GherkinLocalizationTools.\n\n    Provided to keep tests and legacy imports working.\n    '
	def __init__(A,*,decomposition_mode:DecompositionMode=DecompositionMode.GRAMMATICAL,**B):
		if decomposition_mode==DecompositionMode.GHERKIN:A._delegate=GherkinLocalizationTools(**B)
		else:A._delegate=GrammaticalLocalizationTools(**B)
	def all(A)->Tuple[List[BaseTool],List[BaseTool]]:return A._delegate.all()
	def __getattr__(A,item):return getattr(A._delegate,item)
__all__=['BaseLocalizationTools','GrammaticalLocalizationTools','GherkinLocalizationTools','LocalizationTools']