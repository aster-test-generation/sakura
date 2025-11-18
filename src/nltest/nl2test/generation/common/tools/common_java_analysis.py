from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Union

from cldk.models.java.models import JMethodDetail, JCallable
from langchain_core.tools import StructuredTool

from nltest.utils.analysis.common_analysis import CommonAnalysis
from nltest.utils.exceptions import (
    ToolExceptionHandler,
    MethodNotFoundError,
    CallSiteNotFoundError, ClassNotFoundError,
)
from nltest.nl2test.generation.common.tool_descriptions import (
    EXTRACT_CODE_DESC,
    METHOD_DETAILS_DESC,
    CALL_SITE_DETAILS_DESC,
)
from nltest.nl2test.models.agents import QueryMethodArgs, ExtractMethodCodeArgs

from cldk.analysis.java import JavaAnalysis


class CommonJavaAnalysisTools:
    """
    Shared Java static-analysis tools used by both
    localization and composition tool builders.
    """

    def __init__(self, *, analysis: JavaAnalysis) -> None:
        self.analysis = analysis

    # Understand conditional branches and code structure
    def _make_extract_code_tool(self) -> StructuredTool:
        def _extract_method_code(
                qualified_class_name: str,
                method_signature: str,
                start_line: int,
                end_line: int,
        ) -> Dict[str, Any]:
            # Look up method; fail clearly if not found
            class_details = self.analysis.get_class(qualified_class_name)
            if not class_details:
                raise ClassNotFoundError(
                    f"Class {qualified_class_name} not found.",
                    extra_info={"qualified_class_name": qualified_class_name},
                )

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

            # Build the full method source and compute an inclusive slice
            full_source = CommonAnalysis.get_complete_method_code(
                method_details.declaration, method_details.code
            )
            lines = full_source.splitlines()
            total = len(lines)

            # Clamp to 1-based inclusive bounds; return empty when out of range
            start = max(1, start_line)
            end = max(0, end_line)
            if total == 0:
                start = 1
                end = 0
            else:
                start = min(start, total)
                end = min(end, total)

            empty_slice = start > end
            slice_lines: List[str] = [] if empty_slice else lines[start - 1:end]
            note = "" if not empty_slice else "start_line greater than end_line; returning empty slice."

            return {
                "source": "\n".join(slice_lines),
                "start_line": start,
                "end_line": end,
                "total_lines": total,
                "note": note,
            }

        return StructuredTool.from_function(
            func=_extract_method_code,
            name="extract_method_code",
            description=textwrap.dedent(EXTRACT_CODE_DESC).strip(),
            args_schema=ExtractMethodCodeArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    # Get basic method details like what it returns, parameters, modifiers, and comments.
    def _make_get_method_details_tool(self) -> StructuredTool:
        def _get_method_details(
                qualified_class_name: str, method_signature: str
        ) -> Dict[str, any]:
            class_details = self.analysis.get_class(qualified_class_name)
            if not class_details:
                raise ClassNotFoundError(
                    f"Class {qualified_class_name} not found.",
                    extra_info={"qualified_class_name": qualified_class_name},
                )

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
                "comments": [c.content[:25] for c in method_details.comments if c.content],
                "visibility": visibility,
            }

        return StructuredTool.from_function(
            func=_get_method_details,
            name="get_method_details",
            description=textwrap.dedent(METHOD_DETAILS_DESC).strip(),
            args_schema=QueryMethodArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_call_site_details_tool(self) -> StructuredTool:
        def _get_call_site_details(
                qualified_class_name: str, method_signature: str
        ) -> List[Dict[str, Union[str, List[str]]]]:
            class_details = self.analysis.get_class(qualified_class_name)
            if not class_details:
                raise ClassNotFoundError(
                    f"Class {qualified_class_name} not found.",
                    extra_info={"qualified_class_name": qualified_class_name},
                )

            method_details = self.analysis.get_method(
                qualified_class_name, method_signature
            )
            if not method_details:
                raise MethodNotFoundError(
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
                method_details: JCallable = callee_details.method
                lines = entry.get("calling_lines", [])
                count = max(len(lines), 1)
                result.append(
                    {
                        "qualified_class_name": callee_details.klass,
                        "method_signature": method_details.signature,
                        "return_type": method_details.return_type,
                        "parameter_types": [p.type for p in method_details.parameters],
                        "modifiers": method_details.modifiers,
                        "num_times_called": count,
                    }
                )
            return result

        return StructuredTool.from_function(
            func=_get_call_site_details,
            name="get_call_site_details",
            description=textwrap.dedent(CALL_SITE_DETAILS_DESC).strip(),
            args_schema=QueryMethodArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )
