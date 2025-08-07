import json
from typing import List, Dict, Any, Union

from cldk.analysis.java import JavaAnalysis
from cldk.models.java.models import JMethodDetail, JCallable
from langchain_core.tools import StructuredTool, BaseTool

from nltest.nl2test.model.models import AtomicBlock
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.nl2test.preprocessing.searchers import ClassSearcher
from nltest.nl2test.preprocessing.searchers import MethodSearcher
from nltest.utils.analysis import CommonAnalysis, Reachability
from nltest.utils.exceptions import InvalidArgumentError, ToolExceptionHandler, ClassNotFoundError, MethodNotFoundError, \
    CallSiteNotFoundError, FormatError
from nltest.utils.llm import FormatValidator, LLMClient
from nltest.nl2test.generation.localization.tool_descriptions import (QUERY_METHOD_DESC, QUERY_CLASS_DESC,
                                                                      REACHABLE_DESC,
                                                                      EXTRACT_CODE_DESC, METHOD_DETAILS_DESC,
                                                                      CLASS_DETAILS_DESC,
                                                                      CALL_SITE_DETAILS_DESC, MODIFY_BLOCKS_DESC, )


class LocalizationTools:
    def __init__(
            self,
            *,
            analysis: JavaAnalysis,
            method_searcher: MethodSearcher,
            class_searcher: ClassSearcher,
            structured_llm: LLMClient
    ):
        self.analysis = analysis
        self.method_searcher = method_searcher
        self.class_searcher = class_searcher
        self.structured_llm = structured_llm

        self.tools = [
            self._make_query_method_tool(),
            self._make_query_class_tool(),
            self._make_reachable_methods_tool(),
            self._make_extract_code_tool(),
            self._make_method_details_tool(),
            self._make_class_details_tool(),
            self._make_call_site_details_tool(),
            self._make_modify_blocks_tool(),
        ]

    def all(self) -> List[BaseTool]:
        return self.tools

    def _make_query_method_tool(self) -> StructuredTool:
        def _query_method_db(query: str, i: int, j: int) -> List[Dict[str, str]]:
            if i <= 0:
                raise InvalidArgumentError("i must be positive", extra_info={"i": i})
            if j < i:
                raise InvalidArgumentError("j must be greater than i", extra_info={"i": i, "j": j})

            return self.method_searcher.find_similar_in_range(query, i, j)

        return StructuredTool.from_function(
            func=_query_method_db,
            name="query_method_db",
            description=QUERY_METHOD_DESC,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_query_class_tool(self) -> StructuredTool:
        def _query_class_db(query: str, i: int, j: int) -> List[Dict[str, str]]:
            if i <= 0:
                raise InvalidArgumentError("i must be positive", extra_info={"i": i})
            if j < i:
                raise InvalidArgumentError("j must be greater than i", extra_info={"i": i, "j": j})

            return self.class_searcher.find_similar_in_range(query, i, j)

        return StructuredTool.from_function(
            func=_query_class_db,
            name="query_class_db",
            description=QUERY_CLASS_DESC,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_reachable_methods_tool(self) -> StructuredTool:
        def _get_reachable_methods_in_class(qualified_class_name: str) -> Dict[str, List[str]]:
            return Reachability(self.analysis).get_reachable_class_methods(
                qualified_class_name, only_visible=True
            )

        return StructuredTool.from_function(
            func=_get_reachable_methods_in_class,
            name="get_reachable_methods_in_class",
            description=REACHABLE_DESC,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_extract_code_tool(self) -> StructuredTool:
        def _extract_method_code(qualified_class_name: str, method_signature: str) -> str:
            method_details = self.analysis.get_method(qualified_class_name, method_signature)
            if not method_details:
                raise MethodNotFoundError(
                    f"Method {method_signature} not found in class {qualified_class_name}.",
                    extra_info={"qualified_class_name": qualified_class_name, "method_signature": method_signature},
                )

            return CommonAnalysis.get_complete_method_code(method_details.declaration, method_details.code)

        return StructuredTool.from_function(
            func=_extract_method_code,
            name="extract_method_code",
            description=EXTRACT_CODE_DESC,
            handle_tool_error=ToolExceptionHandler.handle_error
        )

    def _make_method_details_tool(self) -> StructuredTool:
        def _get_method_details(qualified_class_name: str, method_signature: str) -> Dict[str, Union[str, List[str]]]:
            method_details = self.analysis.get_method(qualified_class_name, method_signature)
            if not method_details:
                raise MethodNotFoundError(
                    f"Method {method_signature} not found in class {qualified_class_name}.",
                    extra_info={"qualified_class_name": qualified_class_name, "method_signature": method_signature},
                )

            return {
                "method_signature": method_details.signature,
                "modifiers": method_details.modifiers,
                "return_type": method_details.return_type,
                "parameter_types": [p.type for p in method_details.parameters],
                "comments": [c.content for c in method_details.comments],
            }

        return StructuredTool.from_function(
            func=_get_method_details,
            name="get_method_details",
            description=METHOD_DETAILS_DESC,
            handle_tool_error=ToolExceptionHandler.handle_error
        )

    def _make_class_details_tool(self) -> StructuredTool:
        def _get_class_details(qualified_class_name: str) -> Dict[str, Union[str, List[str]]]:
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
                # TODO: Decide if we should get field declarations
            }

        return StructuredTool.from_function(
            func=_get_class_details,
            name="get_class_details",
            description=CLASS_DETAILS_DESC,
            handle_tool_error=ToolExceptionHandler.handle_error
        )

    def _get_inherited_library_classes_tool(self) -> StructuredTool:
        def _get_inherited_library_classes(qualified_class_name: str) -> List[str]:
            pass

    def _make_call_site_details_tool(self) -> StructuredTool:
        def _get_call_site_details(qualified_class_name: str, method_signature: str) -> List[Dict[str, Any]] | str:
            method_details = self.analysis.get_method(qualified_class_name, method_signature)
            if not method_details:
                raise CallSiteNotFoundError(
                    f"Call sites could not be found because method {method_signature} not found in class {qualified_class_name}.",
                    extra_info={"qualified_class_name": qualified_class_name, "method_signature": method_signature},
                )

            entries = self.analysis.get_callees(
                source_class_name=qualified_class_name,
                source_method_declaration=method_signature,
                using_symbol_table=True
            ).get("callee_details", [])

            result: List[Dict[str, Any]] = []
            for entry in entries:
                callee_details: JMethodDetail = entry["callee_method"]
                method_details: JCallable = callee_details.method
                lines = entry.get("calling_lines", [])
                count = max(len(lines), 1)
                result.append({
                    "qualified_class_name": callee_details.klass,
                    "method_signature": method_details.signature,
                    "return_type": method_details.return_type,
                    "parameter_types": [p.type for p in method_details.parameters],
                    "modifiers": method_details.modifiers,
                    "num_times_called": count,
                })
            return result

        return StructuredTool.from_function(
            func=_get_call_site_details,
            name="get_call_site_details",
            description=CALL_SITE_DETAILS_DESC,
            handle_tool_error=ToolExceptionHandler.handle_error
        )

    def _make_modify_blocks_tool(self) -> StructuredTool:
        def _modify_atomic_blocks(current_blocks: List[AtomicBlock], instructions: str) -> List[AtomicBlock]:
            modification_prompt = LoadPrompt.load_prompt("modify_atomic_blocks.jinja2", PromptFormat.JINJA2)
            modification_prompt = modification_prompt.format(
                atomic_blocks=current_blocks,
                instructions=instructions
            )
            result = self.structured_llm.generate(modification_prompt, sanitize=True)

            try:
                return FormatValidator.validate(result, List[AtomicBlock])
            except ValueError as e:
                raise FormatError(
                    f"Failed to validate the formatting of the atomic blocks: {str(e)}",
                    extra_info={"instructions": instructions, "raw_result": result}
                )

        return StructuredTool.from_function(
            func=_modify_atomic_blocks,
            name="modify_atomic_blocks",
            description=MODIFY_BLOCKS_DESC,
            handle_tool_error=ToolExceptionHandler.handle_error
        )

    def _parse_and_validate(self, content: str, expected_type: type) -> Any:
        for _ in range(3):
            try:
                data = json.loads(content)
                if isinstance(data, expected_type):
                    return data
            except json.JSONDecodeError:
                correction = self.structured_llm.generate("TEMP")  # TODO: Modify this prompts
                content = correction
        raise ValueError(f"Failed to parse output as {expected_type}")
