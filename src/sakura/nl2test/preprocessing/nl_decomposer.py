from typing import List, Union

from sakura.nl2test.models.decomposition import (
    GrammaticalBlock,
    GrammaticalBlockList,
    Scenario,
    DecompositionMode,
)
from sakura.nl2test.preprocessing.decomposers import (
    GrammaticalDecomposer,
    GherkinDecomposer,
    BaseDecomposer,
)
from sakura.utils.llm import UsageTracker


class NLDecomposer:
    def __init__(
        self,
        mode: DecompositionMode = DecompositionMode.GRAMMATICAL,
        usage_tracker: UsageTracker | None = None,
    ):
        self.mode = mode
        self.usage_tracker = usage_tracker or UsageTracker()
        impl: BaseDecomposer
        if mode == DecompositionMode.GHERKIN:
            impl = GherkinDecomposer(usage_tracker=self.usage_tracker)
        else:
            impl = GrammaticalDecomposer(usage_tracker=self.usage_tracker)
        self._impl = impl

    def decompose(self, nl_description: str) -> Union[GrammaticalBlockList, Scenario]:
        return self._impl.decompose(nl_description)
