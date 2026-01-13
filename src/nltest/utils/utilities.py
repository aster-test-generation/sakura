from __future__ import annotations

from typing import Optional

from nltest.test2nl.model.models import Test2NLEntry
from nltest.utils.models import AbstractionLevel as NL2Abs
from nltest.utils.models import NL2TestInput


def test2nl_entry_to_nl2test_input(entry: Test2NLEntry) -> NL2TestInput:
    """
    Convert a Test2NLEntry (from Test2NL dataset) to NL2TestInput.
    """

    abs_level: Optional[NL2Abs] = None
    if entry.abstraction_level is not None:
        # Map by enum value ("high"/"medium"/"low")
        try:
            abs_level = NL2Abs(entry.abstraction_level.value)
        except Exception:
            # Fallback: try by name if value mapping fails
            try:
                abs_level = NL2Abs[entry.abstraction_level.name]
            except Exception:
                abs_level = None

    return NL2TestInput(
        id=entry.id,
        description=entry.description,
        project_name=entry.project_name,
        qualified_class_name=entry.qualified_class_name,
        method_signature=entry.method_signature,
        abstraction_level=abs_level,
        is_bdd=entry.is_bdd,
    )


test2nl_entry_to_nl2test_input.__test__ = False
