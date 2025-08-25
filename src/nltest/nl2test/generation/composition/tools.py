import json
from pathlib import Path
from typing import List, Any, Dict, Union

from cldk.analysis.java import JavaAnalysis
from cldk.models.java.models import JMethodDetail, JCallable
from langchain_core.tools import StructuredTool, BaseTool
from requests import HTTPError, JSONDecodeError

from nltest.nl2test.model.models import NL2TestInput, QueryClassArgs, QueryMethodArgs, InstructionArgs, AtomicBlock, ModifyAtomicBlockNotesArgs
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.nl2test.preprocessing.searchers.class_searcher import ClassSearcher
from nltest.nl2test.preprocessing.searchers.method_searcher import MethodSearcher
from nltest.utils.analysis.common_analysis import CommonAnalysis
from nltest.utils.exceptions import CallSiteNotFoundError, ClassNotFoundError, MethodNotFoundError, ToolExceptionHandler, AtomicBlockNotFoundError
from nltest.utils.execution import JavaCompilation
from nltest.utils.execution.execution import JavaExecution
from nltest.utils.file_io.test_file_manager import TestFileManager, TestFileInfo
from nltest.utils.llm import FormatValidator, LLMClient
from nltest.nl2test.generation.composition.tool_descriptions import EXTRACT_CODE_DESC


class CompositionTools:
    def __init__(
            self,
            *,
            analysis: JavaAnalysis,
            method_searcher: MethodSearcher,
            class_searcher: ClassSearcher,
            structured_llm: LLMClient,
            base_project_dir: Union[str, Path],
            nl2_input: NL2TestInput,
    ):
        self.analysis = analysis
        self.method_searcher = method_searcher
        self.class_searcher = class_searcher
        self.structured_llm = structured_llm

        self.project_root: Path = Path(base_project_dir) / nl2_input.project_name
        self.nl2_input: NL2TestInput = nl2_input

        self.tools = [
            self._make_extract_code_tool(),
            self._make_method_details_tool(),
            self._make_view_test_code_tool(),
            self._make_generate_test_tool(),
            self._make_compile_test_tool(),
            self._make_execution_test_tool()
        ]

    def all(self) -> List[BaseTool]:
        return self.tools

    # Understand conditional branches and code structure
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

    def _make_call_site_details_tool(self) -> StructuredTool:
        def _get_call_site_details(qualified_class_name: str, method_signature: str) -> List[
            Dict[str, Union[str, List[str]]]]:
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

            result = []
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
            description="",
            handle_tool_error=ToolExceptionHandler.handle_error
        )

    # For instantiating class properties that might be used
    def _make_get_class_fields_tool(self) -> StructuredTool:
        def _get_class_fields(qualified_class_name: str) -> List[Dict[str, Union[str, List[str]]]]:
            class_details = self.analysis.get_class(qualified_class_name)
            if not class_details:
                raise ClassNotFoundError(
                    f"Class {qualified_class_name} not found.",
                    extra_info={"qualified_class_name": qualified_class_name},
                )

            field_details = []
            for field in class_details.field_declarations:
                field_details.append({
                    "variable_names": field.variables,
                    "type": field.type,
                    "modifiers": field.modifiers,
                })

            return field_details

        return StructuredTool.from_function(
            func=_get_class_fields,
            name="get_class_fields",
            description="",
            args_schema=QueryClassArgs,
            handle_tool_error=ToolExceptionHandler.handle_error
        )

    # For mocking dependencies or imports in test file
    def _make_get_class_imports_tool(self) -> StructuredTool:
        def _get_class_imports(qualified_class_name: str) -> List[str]:
            return CommonAnalysis(self.analysis).get_imports_for_class(qualified_class_name)

        return StructuredTool.from_function(
            func=_get_class_imports,
            name="get_class_imports",
            description="",
            args_schema=QueryClassArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    # Understand how to instantiate class objects and requirements
    def _make_get_class_constructors_and_factories_tool(self) -> StructuredTool:
        def _get_class_constructors_and_factories(qualified_class_name: str) -> List[Dict[str, str]]:
            class_details = self.analysis.get_class(qualified_class_name)
            if not class_details:
                raise ClassNotFoundError(
                    f"Class {qualified_class_name} not found.",
                    extra_info={"qualified_class_name": qualified_class_name},
                )

            class_constructors = []
            class_factories = []

            for method_sig in self.analysis.get_methods_in_class(qualified_class_name):
                method_details = self.analysis.get_method(qualified_class_name, method_sig)
                if not method_details:
                    continue

                if method_details.is_constructor:
                    class_constructors.append(method_sig)
                elif (
                        "static" in method_details.modifiers
                        and qualified_class_name == method_details.return_type
                ):
                    class_factories.append(method_sig)

            class_constructors_and_factories = []
            for method_sig in class_constructors:
                class_constructors_and_factories.append({
                    "method_signature": method_sig,
                    "type": "constructor"
                })
            for method_sig in class_factories:
                class_constructors_and_factories.append({
                    "method_signature": method_sig,
                    "type": "factory"
                })

            return class_constructors_and_factories

        return StructuredTool.from_function(
            func=_get_class_constructors_and_factories,
            name="get_class_constructors_and_factories",
            description="",
            args_schema=QueryClassArgs,
            handle_tool_error=ToolExceptionHandler.handle_error
        )

    def _make_get_getters_and_setters_tool(self) -> StructuredTool:
        def _get_getters_and_setters(qualified_class_name: str) -> List[str]:
            getters_and_setters = []
            for method_sig in self.analysis.get_methods_in_class(qualified_class_name):
                method_details = self.analysis.get_method(qualified_class_name, method_sig)
                if not method_details:
                    continue

                if CommonAnalysis.is_getter_or_setter(method_details):
                    getters_and_setters.append(method_sig)

            return getters_and_setters

        return StructuredTool.from_function(
            func=_get_getters_and_setters,
            name="get_getters_and_setters",
            description="",
            args_schema=QueryClassArgs,
            handle_tool_error=ToolExceptionHandler.handle_error
        )

    def _make_view_test_code_tool(self) -> StructuredTool:
        def _view_test_code() -> str:
            info = TestFileInfo.from_nl2test_input(self.nl2_input)
            raw_code = TestFileManager(self.project_root).load(info)
            return raw_code

        return StructuredTool.from_function(
            func=_view_test_code,
            name="view_test_code",
            description="",
            args_schema=QueryMethodArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_generate_test_tool(self) -> StructuredTool:
        def _generate_test_code(instructions: str) -> str:
            generation_prompt = LoadPrompt.load_prompt("generate_tests.jinja2", PromptFormat.JINJA2)
            generation_prompt = generation_prompt.format(
                instructions=instructions,
            )  # TODO: Refine prompts
            try:
                java_raw = self.structured_llm.generate(generation_prompt, sanitize=True)
                java_code = FormatValidator.strip_java_block(java_raw)

                test_file_info = TestFileInfo.from_nl2test_input(
                    self.nl2_input,
                    test_code=java_code,
                )

                saved_path = TestFileManager(self.project_root).save_single(test_file_info)
                return "Successfully generated test code."
            except HTTPError as e:
                return "Failed to generation test code due to HTTP error."
            except JSONDecodeError as e:
                return "Failed to generation test code due to JSON decode error."

        return StructuredTool.from_function(
            func=_generate_test_code,
            name="generate_test_code",
            description="",
            args_schema=InstructionArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_compile_test_tool(self) -> StructuredTool:
        def _compile_test_code() -> Dict[str, Any]:
            erroneous_classes = JavaCompilation.get_erroneous_classes(self.project_root)

            class_key = TestFileManager(self.project_root).encode_class_name(self.nl2_input.id)
            file_key = f"{class_key}.java"
            has_error = any(ec.endswith(file_key) or ec == file_key for ec in erroneous_classes)

            return {
                "erroneous_classes": erroneous_classes,
                "target_class_file": file_key,
                "has_errors_for_target": has_error,
            }

        return StructuredTool.from_function(
            func=_compile_test_code,
            name="compile_test_code",
            description="",
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    def _make_execution_test_tool(self) -> StructuredTool:
        def _execute_test() -> Dict[str, Any]:
            info = TestFileInfo.from_nl2test_input(self.nl2_input)
            test_fqn = TestFileManager(self.project_root).make_test_fqn(info)
            execution_feedback = JavaExecution.execute(str(self.project_root), test_fqn)
            return execution_feedback

        return StructuredTool.from_function(
            func=_execute_test,
            name="execute_test",
            description="",
            handle_tool_error=ToolExceptionHandler.handle_error,
        )

    # Modify the notes of an atomic block if there is challenges with generating the test case for the block. Eg if the qualified class name is not in the same package as a package-private method.
    def _make_modify_atomic_block_notes_tool(self) -> StructuredTool:
        def _modify_atomic_block_notes(order: int, new_notes: str) -> List[AtomicBlock]:
            atomic_blocks = self.state.atomic_blocks
            if not atomic_blocks or order < 0 or not any(atomic_block.order == order for atomic_block in atomic_blocks):
                raise AtomicBlockNotFoundError(
                    f"Atomic block with order {order} not found.",
                    extra_info={"order": order},
                )

            for block in atomic_blocks:
                if block.order == order:
                    block.notes = new_notes
                    break

            return atomic_blocks

        return StructuredTool.from_function(
            func=_modify_atomic_block_notes,
            name="modify_atomic_block_notes",
            description="",
            args_schema=ModifyAtomicBlockNotesArgs,
            handle_tool_error=ToolExceptionHandler.handle_error,
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
