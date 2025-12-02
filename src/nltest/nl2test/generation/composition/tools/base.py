from __future__ import annotations

from pathlib import Path
import textwrap
from typing import List, Any, Dict, Union, Tuple

from cldk.analysis.java import JavaAnalysis
from langchain_core.tools import StructuredTool, BaseTool

from nltest.nl2test.core.deferred_tool import DeferredTool
from nltest.nl2test.models import (
    NL2TestInput,
    QueryClassArgs,
    GenerateTestCodeArgs,
    FinalizeCommentsArgs,
    NoArgs,
)
from nltest.nl2test.preprocessing.searchers import ClassSearcher, MethodSearcher
from nltest.utils.analysis.common_analysis import CommonAnalysis
from nltest.utils.exceptions import (
    ClassNotFoundError,
    ToolExceptionHandler,
)
from nltest.utils.llm import LLMClient
from nltest.nl2test.generation.composition.tool_descriptions import (
    GET_CLASS_FIELDS_DESC,
    GET_CLASS_IMPORTS_DESC,
    GET_CLASS_CONSTRUCTORS_AND_FACTORIES_DESC,
    GET_GETTERS_AND_SETTERS_DESC,
    GET_MAVEN_DEPENDENCIES_DESC,
    GENERATE_TEST_CODE_DESC,
    FINALIZE_DESC,
)
from nltest.nl2test.generation.common.tools.common_java_analysis import (
    CommonJavaAnalysisTools,
)
from nltest.nl2test.generation.common.tools.common_search import (
    CommonSearchTools,
)
from nltest.nl2test.generation.common.tools.common_test_tools import (
    CommonTestTools,
)
from nltest.utils.file_io.pom_processor import PomProcessor


class BaseCompositionTools(CommonJavaAnalysisTools, CommonSearchTools):
    """
    Shared composition tools; subclasses can extend with mode-specific tools.

    Provides tools for:
    - Code analysis (class fields, imports, constructors, getters/setters)
    - Test generation (generate_test_code - deferred to agent)
    - Test validation (view_test_code, compile_and_execute_test - deferred to agent)
    - Finalization (finalize - deferred to agent)
    """

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

        self.project_root: Path = Path(project_root)
        self.nl2_input: NL2TestInput = nl2_input

        self.tools: List[BaseTool] = [
            self._make_query_class_tool(),
            self._make_extract_code_tool(),
            self._make_get_method_details_tool(),
            self._make_get_class_fields_tool(),
            self._make_get_class_imports_tool(),
            self._make_get_class_constructors_and_factories_tool(),
            self._make_get_getters_and_setters_tool(),
            self._make_get_maven_dependencies_tool(),
            CommonTestTools.make_view_test_code_tool(),
            self._make_generate_test_tool(),
            CommonTestTools.make_compile_and_execute_test_tool(),
            self._make_finalize_tool(),
            self._make_call_site_details_tool(),
        ]

        self.allow_duplicate_tools: List[BaseTool] = [
            CommonTestTools.make_view_test_code_tool(),
            CommonTestTools.make_compile_and_execute_test_tool(),
        ]

    def all(self) -> Tuple[List[BaseTool], List[BaseTool]]:
        return self.tools, self.allow_duplicate_tools

    def _make_get_class_fields_tool(self) -> StructuredTool:
        """Get field declarations for a class."""
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
            description=textwrap.dedent(GET_CLASS_FIELDS_DESC).strip(),
            args_schema=QueryClassArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_get_class_imports_tool(self) -> StructuredTool:
        """Get imports for a class."""
        def _get_class_imports(qualified_class_name: str) -> List[str]:
            class_details = self.analysis.get_class(qualified_class_name)
            if not class_details:
                raise ClassNotFoundError(
                    f"Class {qualified_class_name} not found.",
                    extra_info={"qualified_class_name": qualified_class_name},
                )

            return CommonAnalysis(self.analysis).get_imports_for_class(
                qualified_class_name
            )

        return StructuredTool.from_function(
            func=_get_class_imports,
            name="get_class_imports",
            description=textwrap.dedent(GET_CLASS_IMPORTS_DESC).strip(),
            args_schema=QueryClassArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_get_class_constructors_and_factories_tool(self) -> StructuredTool:
        """Get constructors and factory methods for a class."""
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
            description=textwrap.dedent(
                GET_CLASS_CONSTRUCTORS_AND_FACTORIES_DESC
            ).strip(),
            args_schema=QueryClassArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_get_getters_and_setters_tool(self) -> StructuredTool:
        """Get getter and setter methods for a class."""
        def _get_getters_and_setters(qualified_class_name: str) -> List[str]:
            class_details = self.analysis.get_class(qualified_class_name)
            if not class_details:
                raise ClassNotFoundError(
                    f"Class {qualified_class_name} not found.",
                    extra_info={"qualified_class_name": qualified_class_name},
                )

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
            description=textwrap.dedent(GET_GETTERS_AND_SETTERS_DESC).strip(),
            args_schema=QueryClassArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_get_maven_dependencies_tool(self) -> StructuredTool:
        """Get Maven dependencies from pom.xml."""
        def _get_maven_dependencies() -> List[Dict[str, str]]:
            deps = PomProcessor.identify_dependencies(self.project_root)
            return [
                {"group_id": d.group_id, "artifact_id": d.artifact_id} for d in deps
            ]

        return StructuredTool.from_function(
            func=_get_maven_dependencies,
            name="get_maven_dependencies",
            description=textwrap.dedent(GET_MAVEN_DEPENDENCIES_DESC).strip(),
            args_schema=NoArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_generate_test_tool(self) -> BaseTool:
        """
        Create the generate_test_code tool.

        This is a deferred tool - it returns all inputs and the agent's
        process_tool_output hook handles file saving and state updates.
        """
        return DeferredTool.create(
            name="generate_test_code",
            description=textwrap.dedent(GENERATE_TEST_CODE_DESC).strip(),
            args_schema=GenerateTestCodeArgs,
            returns_input_keys=["test_code", "qualified_class_name", "method_signature"],
            processing_note="Agent saves test file to filesystem and updates state.package, "
                           "state.class_name, state.method_signature",
        )

    def _make_finalize_tool(self) -> BaseTool:
        """
        Create the finalize tool.

        This is a deferred tool - it returns comments and the agent's
        process_tool_output hook sets final state and ends the run.
        """
        return DeferredTool.create(
            name="finalize",
            description=textwrap.dedent(FINALIZE_DESC).strip(),
            args_schema=FinalizeCommentsArgs,
            returns_input_keys=["comments"],
            processing_note="Agent sets state.final_comments, state.finalize_called=True, "
                           "and signals end of execution",
        )
