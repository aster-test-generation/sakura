from __future__ import annotations

import json
from pathlib import Path
from typing import List, Any, Dict, Union, Tuple

from cldk.analysis.java import JavaAnalysis
from cldk.models.java.models import JMethodDetail, JCallable
from langchain_core.tools import StructuredTool, BaseTool

from nltest.nl2test.models import (
    NL2TestInput,
    QueryClassArgs,
    QueryMethodArgs,
    GenerateTestCodeArgs,
    AtomicBlockList,
    FinalizeCommentsArgs,
    NoArgs,
    ViewTestCodeArgs,
)
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.nl2test.preprocessing.searchers import ClassSearcher, MethodSearcher
from nltest.utils.analysis.common_analysis import CommonAnalysis
from nltest.utils.exceptions import (
    CallSiteNotFoundError,
    ClassNotFoundError,
    MethodNotFoundError,
    ToolExceptionHandler,
)
from nltest.utils.execution import JavaCompilation
from nltest.utils.execution.execution import JavaExecution
from nltest.utils.file_io.test_file_manager import TestFileManager, TestFileInfo
from nltest.utils.llm import FormatValidator, LLMClient
from nltest.nl2test.generation.composition.tool_descriptions import (
    GET_CLASS_FIELDS_DESC,
    GET_CLASS_IMPORTS_DESC,
    GET_CLASS_CONSTRUCTORS_AND_FACTORIES_DESC,
    GET_GETTERS_AND_SETTERS_DESC,
    GET_MAVEN_DEPENDENCIES_DESC,
    GENERATE_TEST_CODE_DESC,
    FINALIZE_DESC,
)
from nltest.nl2test.generation.common.tool_descriptions import (
    EXTRACT_CODE_DESC,
    METHOD_DETAILS_DESC,
    CALL_SITE_DETAILS_DESC,
    VIEW_TEST_CODE_DESC,
    COMPILE_AND_EXECUTE_TEST_DESC,
)
from nltest.nl2test.generation.common.tools.common_java_analysis import (
    CommonJavaAnalysisTools,
)
from nltest.nl2test.generation.common.tools.common_search import (
    CommonSearchTools,
)
from nltest.utils.file_io.pom_processor import PomProcessor


class BaseCompositionTools(CommonJavaAnalysisTools, CommonSearchTools):
    """Shared composition tools; subclasses can extend with mode-specific tools."""

    def __init__(
            self,
            *,
            analysis: JavaAnalysis,
            method_searcher: MethodSearcher,
            class_searcher: ClassSearcher,
            structured_llm: LLMClient,
            project_root: str,
            nl2_input: NL2TestInput,
    ) -> None:
        CommonJavaAnalysisTools.__init__(self, analysis=analysis)
        CommonSearchTools.__init__(self, class_searcher=class_searcher)

        self.structured_llm = structured_llm
        self.method_searcher = method_searcher

        # Accept project root directly
        self.project_root: Path = Path(project_root)
        self.nl2_input: NL2TestInput = nl2_input

        # Keep the same initial tool set as before; subclasses may append.
        self.tools: List[BaseTool] = [
            self._make_query_class_tool(),
            self._make_extract_code_tool(),
            self._make_get_method_details_tool(),
            self._make_get_class_fields_tool(),
            self._make_get_class_imports_tool(),
            self._make_get_class_constructors_and_factories_tool(),
            self._make_get_getters_and_setters_tool(),
            self._make_get_maven_dependencies_tool(),
            self._make_view_test_code_tool(),
            self._make_generate_test_tool(),
            self._make_compile_and_execute_test_tool(),
            self._make_finalize_tool(),
            self._make_call_site_details_tool(),
            # self._make_compile_test_tool(),  # DEPRECATED
            # self._make_execution_test_tool(),  # DEPRECATED
        ]

        # Tools that are allowed to be invoked repeatedly with identical args
        # without being treated as duplicates. Empty by default.
        self.allow_duplicate_tools: List[BaseTool] = [
            self._make_view_test_code_tool(),
            self._make_compile_and_execute_test_tool(),
        ]

    def all(self) -> Tuple[List[BaseTool], List[BaseTool]]:
        # Return tool list and the subset allowed to duplicate
        return self.tools, self.allow_duplicate_tools

    # query_class_db now provided by CommonSearchTools

    # For instantiating class properties that might be used
    def _make_get_class_fields_tool(self) -> StructuredTool:
        def _get_class_fields(
                qualified_class_name: str,
        ) -> List[Dict[str, Union[str, List[str]]]]:
            class_details = self.analysis.get_class(qualified_class_name)
            if not class_details:
                raise ClassNotFoundError(
                    f"Class {qualified_class_name} not found.",
                    extra_info={"qualified_class_name": qualified_class_name},
                )

            field_details = []
            for field in class_details.field_declarations:
                field_details.append(
                    {
                        "variable_names": field.variables,
                        "type": field.type,
                        "modifiers": field.modifiers,
                    }
                )

            return field_details

        return StructuredTool.from_function(
            func=_get_class_fields,
            name="get_class_fields",
            description=GET_CLASS_FIELDS_DESC,
            args_schema=QueryClassArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    # For mocking dependencies or imports in test file
    def _make_get_class_imports_tool(self) -> StructuredTool:
        def _get_class_imports(qualified_class_name: str) -> List[str]:
            return CommonAnalysis(self.analysis).get_imports_for_class(
                qualified_class_name
            )

        return StructuredTool.from_function(
            func=_get_class_imports,
            name="get_class_imports",
            description=GET_CLASS_IMPORTS_DESC,
            args_schema=QueryClassArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    # Understand how to instantiate class objects and requirements
    def _make_get_class_constructors_and_factories_tool(self) -> StructuredTool:
        def _get_class_constructors_and_factories(
                qualified_class_name: str,
        ) -> List[Dict[str, str]]:
            class_details = self.analysis.get_class(qualified_class_name)
            if not class_details:
                raise ClassNotFoundError(
                    f"Class {qualified_class_name} not found.",
                    extra_info={"qualified_class_name": qualified_class_name},
                )

            class_constructors: List[str] = []
            class_factories: List[str] = []

            for method_sig in self.analysis.get_methods_in_class(qualified_class_name):
                method_details = self.analysis.get_method(
                    qualified_class_name, method_sig
                )
                if not method_details:
                    continue

                if method_details.is_constructor:
                    class_constructors.append(method_sig)
                elif (
                        "static" in method_details.modifiers
                        and qualified_class_name == method_details.return_type
                ):
                    class_factories.append(method_sig)

            class_constructors_and_factories: List[Dict[str, str]] = []
            for method_sig in class_constructors:
                class_constructors_and_factories.append(
                    {"method_signature": method_sig, "type": "constructor"}
                )
            for method_sig in class_factories:
                class_constructors_and_factories.append(
                    {"method_signature": method_sig, "type": "factory"}
                )

            return class_constructors_and_factories

        return StructuredTool.from_function(
            func=_get_class_constructors_and_factories,
            name="get_class_constructors_and_factories",
            description=GET_CLASS_CONSTRUCTORS_AND_FACTORIES_DESC,
            args_schema=QueryClassArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_get_getters_and_setters_tool(self) -> StructuredTool:
        def _get_getters_and_setters(qualified_class_name: str) -> List[str]:
            getters_and_setters: List[str] = []
            for method_sig in self.analysis.get_methods_in_class(qualified_class_name):
                method_details = self.analysis.get_method(
                    qualified_class_name, method_sig
                )
                if not method_details:
                    continue

                if CommonAnalysis.is_getter_or_setter(method_details):
                    getters_and_setters.append(method_sig)

            return getters_and_setters

        return StructuredTool.from_function(
            func=_get_getters_and_setters,
            name="get_getters_and_setters",
            description=GET_GETTERS_AND_SETTERS_DESC,
            args_schema=QueryClassArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_get_maven_dependencies_tool(self) -> StructuredTool:
        def _get_maven_dependencies() -> List[Dict[str, str]]:
            deps = PomProcessor.identify_dependencies(self.project_root)
            return [
                {"group_id": d.group_id, "artifact_id": d.artifact_id} for d in deps
            ]

        return StructuredTool.from_function(
            func=_get_maven_dependencies,
            name="get_maven_dependencies",
            description=GET_MAVEN_DEPENDENCIES_DESC,
            args_schema=NoArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_view_test_code_tool(self) -> StructuredTool:
        def _view_test_code(start_line: int, end_line: int) -> dict:
            if start_line <= 1 or end_line <= 1:
                raise ValueError("start_line and end_line must be greater than 1.")
            if end_line < start_line:
                raise ValueError("end_line must be greater than or equal to start_line.")
            return {"start_line": start_line, "end_line": end_line}

        return StructuredTool.from_function(
            func=_view_test_code,
            name="view_test_code",
            description=VIEW_TEST_CODE_DESC,
            args_schema=ViewTestCodeArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_generate_test_tool(self) -> StructuredTool:
        def _generate_test_code(
                test_code: str, qualified_class_name: str, method_signature: str
        ) -> dict:
            # NOTE: Work is done by the agent hook for state injection
            return {
                "test_code": test_code,
                "qualified_class_name": qualified_class_name,
                "method_signature": method_signature,
            }

        return StructuredTool.from_function(
            func=_generate_test_code,
            name="generate_test_code",
            description=GENERATE_TEST_CODE_DESC,
            args_schema=GenerateTestCodeArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    # DEPRECATED
    def _make_compile_test_tool(self) -> StructuredTool:
        def _compile_test_code() -> Dict[str, Any]:
            erroneous_files = JavaCompilation.get_erroneous_files(self.project_root)

            class_key = TestFileManager(self.project_root).encode_class_name(
                self.nl2_input.id
            )
            file_key = f"{class_key}.java"
            has_error = any(
                ef.endswith(file_key) or ef == file_key for ef in erroneous_files
            )

            return {
                "erroneous_files": erroneous_files,
                "target_class_file": file_key,
                "has_errors_for_target": has_error,
            }

        return StructuredTool.from_function(
            func=_compile_test_code,
            name="compile_test_code",
            description="",
            args_schema=NoArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_compile_and_execute_test_tool(self) -> StructuredTool:
        def _compile_and_execute_test() -> dict:
            # NOTE: Work is done by the agent for state injection
            return {}

        return StructuredTool.from_function(
            func=_compile_and_execute_test,
            name="compile_and_execute_test",
            description=COMPILE_AND_EXECUTE_TEST_DESC,
            args_schema=NoArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_finalize_tool(self) -> StructuredTool:
        def _finalize(comments: str) -> str:
            # Return comments; agent will set final state and end.
            return str(comments)

        return StructuredTool.from_function(
            func=_finalize,
            name="finalize",
            description=FINALIZE_DESC,
            args_schema=FinalizeCommentsArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    # Removed legacy _parse_and_validate helper; structured tools return typed outputs.
