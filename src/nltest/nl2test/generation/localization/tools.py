import json
from typing import List, Dict, Any, Union, Literal, Tuple

from cldk.analysis.java import JavaAnalysis
from cldk.models.java.models import JMethodDetail, JCallable
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import StructuredTool, BaseTool
from pydantic import RootModel, BaseModel, Field

from nltest.nl2test.model.models import AtomicBlock, QueryMethodArgs, QueryClassArgs, QueryVectorDataArgs, \
    ReachableMethodsArgs, InstructionArgs, FinalizeBlocksArgs
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.nl2test.preprocessing.searchers import ClassSearcher
from nltest.nl2test.preprocessing.searchers import MethodSearcher
from nltest.utils.analysis import CommonAnalysis, Reachability
from nltest.utils.exceptions import InvalidArgumentError, ToolExceptionHandler, ClassNotFoundError, MethodNotFoundError, \
    CallSiteNotFoundError, FormatError
from nltest.utils.exceptions.tool_exceptions import AtomicBlockNotFoundError
from nltest.utils.llm import FormatValidator, LLMClient
from nltest.nl2test.generation.localization.tool_descriptions import (QUERY_METHOD_DESC, QUERY_CLASS_DESC,
                                                                      REACHABLE_METHODS_DESC,
                                                                      EXTRACT_CODE_DESC, METHOD_DETAILS_DESC,
                                                                      CLASS_DETAILS_DESC, INHERITED_LIBRARY_CLASSES_DESC,
                                                                      CALL_SITE_DETAILS_DESC, MODIFY_BLOCKS_DESC,
                                                                      FINALIZE_BLOCKS_DESC, )

class AtomicBlockList(BaseModel):
    atomic_blocks: List[AtomicBlock] = Field(..., description="List of atomic blocks.")


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
            self._make_get_inherited_library_classes_tool(),
            # self._make_call_site_details_tool(),
            self._make_modify_blocks_tool(),
            self._make_finalize_atomic_blocks_tool(),
        ]

    def all(self) -> List[BaseTool]:
        return self.tools

    # Get relevant methods from the database by similarity search, within a range
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
            args_schema=QueryVectorDataArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    # Get relevant classes from the database by similarity search, within a range
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
            args_schema=QueryVectorDataArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    # Get all the methods that can be called from the class, looking at its inheritance graph
    def _make_reachable_methods_tool(self) -> StructuredTool:
        def _get_reachable_methods_in_class(
                qualified_class_name: str,
                visibility_mode: str
        ) -> Dict[str, List[Dict[str, Any]]]:
            if visibility_mode not in ("public", "same_package", "same_package_or_subclass"):
                raise InvalidArgumentError(
                    "Invalid visibility mode", extra_info={"visibility_mode": visibility_mode}
                )

            return Reachability(self.analysis).get_visible_class_methods(
                qualified_class_name,
                visibility_mode=visibility_mode,
                include_metadata=True
            )

        return StructuredTool.from_function(
            func=_get_reachable_methods_in_class,
            name="get_reachable_methods_in_class",
            description=REACHABLE_METHODS_DESC,
            args_schema=ReachableMethodsArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    # Get the complete method code
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
            args_schema=QueryMethodArgs,
            handle_tool_error=ToolExceptionHandler.handle_error
        )

    # Get basic method details like what it returns, parameters, modifiers, and comments.
    def _make_method_details_tool(self) -> StructuredTool: 
        def _get_method_details(qualified_class_name: str, method_signature: str) -> Dict[str, Union[str, List[str]]]:
            method_details = self.analysis.get_method(qualified_class_name, method_signature)
            if not method_details:
                raise MethodNotFoundError(
                    f"Method {method_signature} not found in class {qualified_class_name}.",
                    extra_info={"qualified_class_name": qualified_class_name, "method_signature": method_signature},
                )

            common_analysis = CommonAnalysis(self.analysis)
            visibility = common_analysis.get_method_visibility(qualified_class_name, method_signature)

            return {
                "method_signature": method_details.signature,
                "modifiers": method_details.modifiers,
                "return_type": method_details.return_type,
                "parameter_types": [p.type for p in method_details.parameters],
                "comments": [c.content for c in method_details.comments],
                "visibility": visibility,  # Options: "public", "same_package_or_subclass", "same_package"
            }

        return StructuredTool.from_function(
            func=_get_method_details,
            name="get_method_details",
            description=METHOD_DETAILS_DESC,
            args_schema=QueryMethodArgs,
            handle_tool_error=ToolExceptionHandler.handle_error
        )

    # Get basic class details like what it extends, implements, modifiers, and annotations. 
    # Only includes information from the class itself, not its inheritance graph.
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
            args_schema=QueryClassArgs,
            handle_tool_error=ToolExceptionHandler.handle_error
        )

    # Get the inherited library classes to get any library methods that cannot be found through static analysis on the application
    def _make_get_inherited_library_classes_tool(self) -> StructuredTool:
        reachability = Reachability(self.analysis)

        def _get_inherited_library_classes(qualified_class_name: str) -> List[str]:
            inherited = reachability.get_inherited_classes_and_interfaces(qualified_class_name)

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
            description=INHERITED_LIBRARY_CLASSES_DESC,
            args_schema=QueryClassArgs,
            handle_tool_error=ToolExceptionHandler.handle_error
        )

    # Get the call site details
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
            args_schema=QueryMethodArgs,
            handle_tool_error=ToolExceptionHandler.handle_error
        )

    # Make modifications to the atomic blocks based on static analysis
    def _make_modify_blocks_tool(self) -> StructuredTool:
        def _modify_atomic_blocks(instructions: str, current_blocks: List[AtomicBlock] = None) -> List[AtomicBlock]:
            # NOTE: current_blocks is passed in as an argument from the agent state
            if current_blocks is None:
                raise AtomicBlockNotFoundError("Current blocks not found", extra_info={"current_blocks": current_blocks})

            modification_system = LoadPrompt.load_prompt("modify_atomic_blocks.jinja2", PromptFormat.JINJA2, prompt_type="system").format()
            modification_chat = LoadPrompt.load_prompt("modify_atomic_blocks.jinja2", PromptFormat.JINJA2, prompt_type="chat").format(
                atomic_blocks=current_blocks,
                instructions=instructions
            )

            result: AtomicBlockList = self.structured_llm.invoke_prompts(
                modification_system,
                modification_chat,
                schema=AtomicBlockList,
                strict=True,
            )

            return result.atomic_blocks

        return StructuredTool.from_function(
            func=_modify_atomic_blocks,
            name="modify_atomic_blocks",
            description=MODIFY_BLOCKS_DESC,
            args_schema=InstructionArgs,
            handle_tool_error=ToolExceptionHandler.handle_error
        )

    # Finalize the atomic blocks and end the agent
    def _make_finalize_atomic_blocks_tool(self) -> StructuredTool:
        def _finalize_atomic_blocks(comments: str, current_blocks: List[AtomicBlock] = None) -> Tuple[List[AtomicBlock], str]:
            # NOTE: current_blocks is passed in as an argument from the agent state
            if current_blocks is None:
                raise AtomicBlockNotFoundError("Current blocks not found", extra_info={"current_blocks": current_blocks})

            return current_blocks, comments

        return StructuredTool.from_function(
            func=_finalize_atomic_blocks,
            name="finalize_atomic_blocks",
            description=FINALIZE_BLOCKS_DESC,
            args_schema=FinalizeBlocksArgs,
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