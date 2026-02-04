from __future__ import annotations
from typing import Optional
from sakura.test2nl.model.models import Test2NLEntry
from sakura.utils.models import AbstractionLevel as NL2Abs
from sakura.utils.models import NL2TestInput
def test2nl_entry_to_nl2test_input(entry:Test2NLEntry)->NL2TestInput:
	'\n    Convert a Test2NLEntry (from Test2NL dataset) to NL2TestInput.\n    ';C=None;A=entry;B:Optional[NL2Abs]=C
	if A.abstraction_level is not C:
		try:B=NL2Abs(A.abstraction_level.value)
		except Exception:
			try:B=NL2Abs[A.abstraction_level.name]
			except Exception:B=C
	return NL2TestInput(id=A.id,description=A.description,project_name=A.project_name,qualified_class_name=A.qualified_class_name,method_signature=A.method_signature,abstraction_level=B,is_bdd=A.is_bdd)
test2nl_entry_to_nl2test_input.__test__=False