from __future__ import annotations

from typing import List, Dict

from langchain_core.tools import StructuredTool

from nltest.utils.exceptions import InvalidArgumentError, ToolExceptionHandler
from nltest.nl2test.models import QueryVectorDataArgs
from nltest.nl2test.generation.common.tool_descriptions import QUERY_CLASS_DESC


class CommonSearchTools:
    """Shared vector-search tools for class/method discovery.

    Expects subclasses to provide a `class_searcher` capable of
    `find_similar_in_range(query, i, j)`.
    """

    def __init__(self, *, class_searcher) -> None:
        self.class_searcher = class_searcher

    def _make_query_class_tool(self) -> StructuredTool:
        def _query_class_db(query: str, i: int, j: int) -> List[Dict[str, str]]:
            if i <= 0:
                raise InvalidArgumentError("i must be positive", extra_info={"i": i})
            if j < i:
                raise InvalidArgumentError(
                    "j must be greater than i", extra_info={"i": i, "j": j}
                )
            return self.class_searcher.find_similar_in_range(query, i, j)

        return StructuredTool.from_function(
            func=_query_class_db,
            name="query_class_db",
            description=QUERY_CLASS_DESC,
            args_schema=QueryVectorDataArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )
