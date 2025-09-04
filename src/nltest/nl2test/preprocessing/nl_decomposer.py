from typing import List, Union

from nltest.nl2test.models.decomposition import (
    GrammaticalBlock,
    GrammaticalBlockList,
    Scenario,
    DecompositionMode,
)
from nltest.nl2test.preprocessing.decomposers import (
    GrammaticalDecomposer,
    GherkinDecomposer,
    BaseDecomposer,
)


class NLDecomposer:
    def __init__(self, mode: DecompositionMode = DecompositionMode.GRAMMATICAL):
        self.mode = mode
        impl: BaseDecomposer
        if mode == DecompositionMode.GHERKIN:
            impl = GherkinDecomposer()
        else:
            impl = GrammaticalDecomposer()
        self._impl = impl

    def decompose(self, nl_description: str) -> Union[GrammaticalBlockList, Scenario]:
        return self._impl.decompose(nl_description)
