import textwrap
from typing import Any, Dict, List, Tuple, Union

from cldk.analysis.java import JavaAnalysis
from langchain_core.tools import BaseTool, StructuredTool

from nltest.nl2test.generation.common.tools.common_java_analysis import (
    CommonJavaAnalysisTools,
)
from nltest.nl2test.generation.common.tools.common_search import (
    CommonSearchTools,
)
from nltest.nl2test.generation.localization.tool_descriptions import (
    CLASS_DETAILS_DESC,
    INHERITED_LIBRARY_CLASSES_DESC,
    QUERY_METHOD_DESC,
    SEARCH_REACHABLE_METHODS_DESC,
)
from nltest.nl2test.models import (
    QueryClassArgs,
    QueryVectorDataArgs,
    SearchReachableMethodsArgs,
)
from nltest.nl2test.preprocessing.searchers import ClassSearcher, MethodSearcher
from nltest.utils.analysis import Reachability
from nltest.utils.exceptions import (
    ClassNotFoundError,
    InvalidArgumentError,
    ToolExceptionHandler,
)


class BaseLocalizationTools(CommonJavaAnalysisTools, CommonSearchTools):
    """Shared localization tools; subclasses implement finalize step."""

    def __init__(
        self,
        *,
        analysis: JavaAnalysis,
        method_searcher: MethodSearcher,
        class_searcher: ClassSearcher,
    ) -> None:
        CommonJavaAnalysisTools.__init__(self, analysis=analysis)
        CommonSearchTools.__init__(self, class_searcher=class_searcher)

        self.method_searcher = method_searcher

        # Subclasses should add their finalize tool if desired
        self.tools: List[BaseTool] = [
            self._make_query_method_tool(),
            self._make_query_class_tool(),
            self._make_search_reachable_methods_tool(),
            self._make_extract_code_tool(),
            self._make_get_method_details_tool(),
            # self._make_get_class_details_tool(),
            self._make_get_inherited_library_classes_tool(),
            self._make_call_site_details_tool(),
        ]

        # Allows tools
        self.allow_duplicate_tools: List[BaseTool] = []

    def all(self) -> Tuple[List[BaseTool], List[BaseTool]]:
        # Return tool list and the subset allowed to duplicate
        return self.tools, self.allow_duplicate_tools

    # Get relevant methods from the database by similarity search, within a range
    def _make_query_method_tool(self) -> StructuredTool:
        def _query_method_db(query: str, i: int, j: int) -> List[Dict[str, str]]:
            if i <= 0:
                raise InvalidArgumentError("i must be positive", extra_info={"i": i})
            if j < i:
                raise InvalidArgumentError(
                    "j must be greater than i", extra_info={"i": i, "j": j}
                )

            return self.method_searcher.find_similar_in_range(query, i, j)

        return StructuredTool.from_function(
            func=_query_method_db,
            name="query_method_db",
            description=textwrap.dedent(QUERY_METHOD_DESC).strip(),
            args_schema=QueryVectorDataArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    # Search reachable methods from the class using semantic similarity.
    def _make_search_reachable_methods_tool(self) -> StructuredTool:
        def _search_reachable_methods_in_class(
            qualified_class_name: str,
            query: str,
            visibility_mode: str,
            k: int = 5,
        ) -> List[Dict[str, Any]]:
            if visibility_mode not in (
                "public",
                "same_package",
                "same_package_or_subclass",
            ):
                raise InvalidArgumentError(
                    "Invalid visibility mode",
                    extra_info={"visibility_mode": visibility_mode},
                )
            if k <= 0:
                raise InvalidArgumentError("k must be positive", extra_info={"k": k})

            oversample_factor = 10
            fetch_k = max(k * oversample_factor, k)
            search_hits = self.method_searcher.find_similar(query, k=fetch_k)

            reachable = Reachability(self.analysis).get_visible_class_methods(
                qualified_class_name,
                visibility_mode=visibility_mode,
                include_metadata=True,
            )

            reachable_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for owner, methods in reachable.items():
                for meta in methods:
                    if not isinstance(meta, dict):
                        continue
                    method_sig = meta.get("method_signature")
                    if not method_sig:
                        continue
                    reachable_lookup[(owner, method_sig)] = meta

            results: List[Dict[str, Any]] = []
            seen: set[Tuple[str, str]] = set()

            for hit in search_hits:
                if hit.get("containing_class_name") != qualified_class_name:
                    continue
                declaring_class = hit.get("declaring_class_name")
                method_sig = hit.get("method_signature")
                if not declaring_class or not method_sig:
                    continue
                key = (declaring_class, method_sig)
                if key in seen:
                    continue
                meta = reachable_lookup.get(key)
                if not meta:
                    continue
                seen.add(key)
                results.append(
                    {
                        "method_signature": method_sig,
                        "declaring_class_name": declaring_class,
                        "containing_class_name": qualified_class_name,
                        "modifiers": meta.get("modifiers", []),
                        "visibility": meta.get("visibility"),
                        "requires_subclass": meta.get("requires_subclass", False),
                    }
                )
                if len(results) >= k:
                    break

            return results

        return StructuredTool.from_function(
            func=_search_reachable_methods_in_class,
            name="search_reachable_methods_in_class",
            description=textwrap.dedent(SEARCH_REACHABLE_METHODS_DESC).strip(),
            args_schema=SearchReachableMethodsArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    # Get basic class details like what it extends, implements, modifiers, and annotations.
    def _make_get_class_details_tool(self) -> StructuredTool:
        def _get_class_details(
            qualified_class_name: str,
        ) -> Dict[str, Union[str, List[str], None]]:
            class_details = self.analysis.get_class(qualified_class_name)
            if not class_details:
                raise ClassNotFoundError(
                    f"Class {qualified_class_name} not found.",
                    extra_info={"qualified_class_name": qualified_class_name},
                )

            return {
                "class_name": qualified_class_name.split(".")[-1],
                "modifiers": class_details.modifiers,
                "extends_list": class_details.extends_list,
                "implements_list": class_details.implements_list,
                "annotations": class_details.annotations,
            }

        return StructuredTool.from_function(
            func=_get_class_details,
            name="get_class_details",
            description=textwrap.dedent(CLASS_DETAILS_DESC).strip(),
            args_schema=QueryClassArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    # Get the inherited library classes to get any library methods that cannot be found through static analysis on the application
    def _make_get_inherited_library_classes_tool(self) -> StructuredTool:
        reachability = Reachability(self.analysis)

        def _get_inherited_library_classes(qualified_class_name: str) -> List[str]:
            class_details = self.analysis.get_class(qualified_class_name)
            if not class_details:
                raise ClassNotFoundError(
                    f"Class {qualified_class_name} not found.",
                    extra_info={"qualified_class_name": qualified_class_name},
                )

            inherited = reachability.get_inherited_classes_and_interfaces(
                qualified_class_name
            )

            seen: set[str] = set()
            out: List[str] = []
            for cls in inherited:
                if cls in seen:
                    continue
                seen.add(cls)

                # "Library" classes considered as those not in application
                if not self.analysis.get_class(cls):
                    out.append(cls)

            return out

        return StructuredTool.from_function(
            func=_get_inherited_library_classes,
            name="get_inherited_library_classes",
            description=textwrap.dedent(INHERITED_LIBRARY_CLASSES_DESC).strip(),
            args_schema=QueryClassArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )
