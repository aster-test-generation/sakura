from __future__ import annotations

from .base import BaseCompositionTools
from langchain_core.tools import StructuredTool

from nltest.nl2test.models import AtomicBlockList, ModifyAtomicBlocksArgs
from nltest.nl2test.models.agents import ModifyAtomicBlockNoteArgs
from nltest.utils.exceptions import ToolExceptionHandler


class GrammaticalCompositionTools(BaseCompositionTools):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
            description="",
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
            description="",
            args_schema=ModifyAtomicBlockNoteArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )
