from __future__ import annotations

from typing import Any, Dict, List, Union

from cldk.models.java.models import JMethodDetail, JCallable
from langchain_core.tools import StructuredTool

from nltest.utils.analysis.common_analysis import CommonAnalysis
from nltest.utils.exceptions import (
    ToolExceptionHandler,
    MethodNotFoundError,
    CallSiteNotFoundError,
)
from nltest.nl2test.generation.common.tool_descriptions import (
    EXTRACT_CODE_DESC,
    METHOD_DETAILS_DESC,
    CALL_SITE_DETAILS_DESC,
)
from cldk.analysis.java import JavaAnalysis


class CommonJavaAnalysisToolsMixin:
    """
    Mixin providing shared Java static-analysis tools used by both
    localization and composition tool builders.
    """

    def __init__(self, *, analysis: JavaAnalysis) -> None:
        self.analysis = analysis

    def _make_extract_code_tool(self) -> StructuredTool:
        def _extract_method_code(
            qualified_class_name: str, method_signature: str
        ) -> str:
            method_details = self.analysis.get_method(
                qualified_class_name, method_signature
            )
            if not method_details:
                raise MethodNotFoundError(
                    f"Method {method_signature} not found in class {qualified_class_name}.",
                    extra_info={
                        "qualified_class_name": qualified_class_name,
                        "method_signature": method_signature,
                    },
                )

            return CommonAnalysis.get_complete_method_code(
                method_details.declaration, method_details.code
            )

        return StructuredTool.from_function(
            func=_extract_method_code,
            name="extract_method_code",
            description=EXTRACT_CODE_DESC,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_method_details_tool(self) -> StructuredTool:
        def _get_method_details(
            qualified_class_name: str, method_signature: str
        ) -> Dict[str, Union[str, List[str]]]:
            method_details = self.analysis.get_method(
                qualified_class_name, method_signature
            )
            if not method_details:
                raise MethodNotFoundError(
                    f"Method {method_signature} not found in class {qualified_class_name}.",
                    extra_info={
                        "qualified_class_name": qualified_class_name,
                        "method_signature": method_signature,
                    },
                )

            common_analysis = CommonAnalysis(self.analysis)
            visibility = common_analysis.get_method_visibility(
                qualified_class_name, method_signature
            )

            return {
                "method_signature": method_details.signature,
                "modifiers": method_details.modifiers,
                "return_type": method_details.return_type,
                "parameter_types": [p.type for p in method_details.parameters],
                "comments": [c.content for c in method_details.comments],
                "visibility": visibility,
            }

        return StructuredTool.from_function(
            func=_get_method_details,
            name="get_method_details",
            description=METHOD_DETAILS_DESC,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_call_site_details_tool(self) -> StructuredTool:
        def _get_call_site_details(
            qualified_class_name: str, method_signature: str
        ) -> List[Dict[str, Any]]:
            method_details = self.analysis.get_method(
                qualified_class_name, method_signature
            )
            if not method_details:
                raise CallSiteNotFoundError(
                    f"Call sites could not be found because method {method_signature} not found in class {qualified_class_name}.",
                    extra_info={
                        "qualified_class_name": qualified_class_name,
                        "method_signature": method_signature,
                    },
                )

            entries = self.analysis.get_callees(
                source_class_name=qualified_class_name,
                source_method_declaration=method_signature,
                using_symbol_table=True,
            ).get("callee_details", [])

            result: List[Dict[str, Any]] = []
            for entry in entries:
                callee_details: JMethodDetail = entry["callee_method"]
                method: JCallable = callee_details.method
                lines = entry.get("calling_lines", [])
                count = max(len(lines), 1)
                result.append(
                    {
                        "qualified_class_name": callee_details.klass,
                        "method_signature": method.signature,
                        "return_type": method.return_type,
                        "parameter_types": [p.type for p in method.parameters],
                        "modifiers": method.modifiers,
                        "num_times_called": count,
                    }
                )
            return result

        return StructuredTool.from_function(
            func=_get_call_site_details,
            name="get_call_site_details",
            description=CALL_SITE_DETAILS_DESC,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )
