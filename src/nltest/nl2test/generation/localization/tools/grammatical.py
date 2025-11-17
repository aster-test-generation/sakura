import textwrap
from typing import Tuple
from cldk.analysis.java import JavaAnalysis
from nltest.nl2test.preprocessing.searchers import MethodSearcher, ClassSearcher

from langchain_core.tools import StructuredTool

from nltest.nl2test.generation.localization.tool_descriptions import (
    FINALIZE_ATOMIC_BLOCKS_DESC,
)
from nltest.nl2test.models import AtomicBlockList, FinalizeAtomicBlockArgs
from nltest.utils.exceptions import ToolExceptionHandler
from nltest.utils.exceptions.tool_exceptions import BlockNotFoundError

from .base import BaseLocalizationTools


class GrammaticalLocalizationTools(BaseLocalizationTools):
    def __init__(
        self,
        *,
        analysis: JavaAnalysis,
        method_searcher: MethodSearcher,
        class_searcher: ClassSearcher,
    ) -> None:
        super().__init__(
            analysis=analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
        )
        # Add the grammatical finalize tool
        self.tools.append(self._make_finalize_tool())

    # Finalize the atomic blocks and end the agent (Grammatical mode)
    def _make_finalize_tool(self) -> StructuredTool:
        def _finalize(
            current_blocks: AtomicBlockList, comments: str
        ) -> Tuple[AtomicBlockList, str]:
            # NOTE: current_blocks is passed in as an argument from the agent state
            if current_blocks is None:
                raise BlockNotFoundError(
                    "Current blocks not found",
                    extra_info={"current_blocks": current_blocks},
                )

            return current_blocks, comments

        return StructuredTool.from_function(
            func=_finalize,
            name="finalize",
            description=textwrap.dedent(FINALIZE_ATOMIC_BLOCKS_DESC).strip(),
            args_schema=FinalizeAtomicBlockArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )
