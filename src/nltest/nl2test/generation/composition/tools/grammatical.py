from __future__ import annotations

from pathlib import Path
from typing import Union
from langchain_core.tools import StructuredTool

from nltest.nl2test.models import (
    AtomicBlockList,
    ModifyAtomicBlocksArgs,
    ModifyAtomicBlockNoteArgs,
)
from nltest.utils.exceptions import ToolExceptionHandler

from .base import BaseCompositionTools
from cldk.analysis.java import JavaAnalysis
from nltest.nl2test.preprocessing.searchers import MethodSearcher, ClassSearcher
from nltest.utils.llm import LLMClient
from nltest.nl2test.models import NL2TestInput
from nltest.nl2test.generation.composition.tool_descriptions import (
    MODIFY_ATOMIC_BLOCKS_DESC,
    MODIFY_ATOMIC_BLOCK_NOTE_DESC,
)


class GrammaticalCompositionTools(BaseCompositionTools):
    def __init__(
        self,
        *,
        analysis: JavaAnalysis,
        method_searcher: MethodSearcher,
        class_searcher: ClassSearcher,
        structured_llm: LLMClient,
        project_root: Union[str, Path],
        nl2_input: NL2TestInput,
    ):
        super().__init__(
            analysis=analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            structured_llm=structured_llm,
            project_root=str(project_root),
            nl2_input=nl2_input,
        )
        # Add grammatical-mode-specific tool(s)
        self.tools.append(self._make_modify_atomic_blocks_tool())
        self.tools.append(self._make_modify_scenario_comment_tool())

    def _make_modify_atomic_blocks_tool(self) -> StructuredTool:
        def _modify_atomic_blocks(atomic_blocks: AtomicBlockList) -> AtomicBlockList:
            # Simply return the atomic blocks as-is per requirements
            return atomic_blocks

        return StructuredTool.from_function(
            func=_modify_atomic_blocks,
            name="modify_atomic_blocks",
            description=MODIFY_ATOMIC_BLOCKS_DESC,
            args_schema=ModifyAtomicBlocksArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_modify_scenario_comment_tool(self) -> StructuredTool:
        def _modify_scenario_comment(order: int, note: str):
            # Directly return the provided values; agent applies the update
            return order, note

        return StructuredTool.from_function(
            func=_modify_scenario_comment,
            name="modify_scenario_comment",
            description=MODIFY_ATOMIC_BLOCK_NOTE_DESC,
            args_schema=ModifyAtomicBlockNoteArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )
