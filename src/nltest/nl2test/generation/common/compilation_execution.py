from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import ToolMessage, ToolCall

from nltest.nl2test.models import AgentState
from nltest.utils.compilation.maven import CompilationError, JavaMavenCompilation
from nltest.utils.execution.maven import ExecutionIssue, JavaMavenExecution
from nltest.utils.formatting import ErrorFormatter
from nltest.utils.tool_messages import format_tool_error, format_tool_ok


class CompilationExecutionMixin:
    """
    Shared mixin providing compile_and_execute_test processing logic.

    Requirements:
    - The using class must have a `project_root` attribute (Path or None)
    - Optional `module_root` attribute for multi-module builds
    """

    project_root: Path | None
    module_root: Path | None = None

    def process_compile_and_execute(
        self,
        tool_call: ToolCall,
        state: AgentState,
        outputs: List[ToolMessage],
    ) -> None:
        """
        Process compile_and_execute_test tool output.

        Runs Maven compilation, checks for errors in target class,
        runs test execution if compilation succeeds, and builds result payload.

        Args:
            tool_call: The tool call being processed
            state: Current agent state with class_name, package, method_signature
            outputs: List to append ToolMessage results to

        Notes:
            If unrelated project files fail to compile, this tool reports a structured
            error payload rather than raising an exception.
        """
        tool_name = tool_call["name"]

        if not state.class_name or not state.method_signature:
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code="no_active_test",
                        message="No test code has been generated or saved.",
                        details={"tool": tool_name, "tool_call_id": tool_call["id"]},
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        if self.project_root is None:
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code="no_project_root",
                        message=(
                            "Project root is not configured; cannot compile or execute tests."
                        ),
                        details={"tool": tool_name, "tool_call_id": tool_call["id"]},
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        project_root = self.project_root
        module_root = self.module_root
        compilation_errors: List[CompilationError] = JavaMavenCompilation(
            project_root, module_root=module_root
        ).get_compilation_errors()

        file_key = f"{state.class_name}.java"
        file_rel_path = (
            f"{state.package.replace('.', '/')}/{file_key}"
            if state.package
            else file_key
        )

        def _matches_error_path(err: str) -> bool:
            normalized = err.replace("\\", "/")
            if "/" in normalized:
                return normalized.endswith(file_rel_path)
            return normalized.endswith(file_key)

        has_error_for_project = len(compilation_errors) > 0
        has_error_for_target = any(
            _matches_error_path(ef.file) for ef in compilation_errors
        )

        target_errors: List[str] = [
            ErrorFormatter.format_compilation_error(ce)
            for ce in compilation_errors
            if _matches_error_path(ce.file)
        ]

        result_payload: Dict[str, Any] = {
            "compilation": {
                "status": "success"
                if not has_error_for_project
                else "compilation_error",
                "target_class_file": file_key,
                "has_errors_for_project": has_error_for_project,
                "has_errors_for_target": has_error_for_target,
                "error_details_for_target_class": target_errors,
            }
        }

        if has_error_for_project and not has_error_for_target:
            files_with_errors = [ce.file for ce in compilation_errors]
            error_details = [
                ErrorFormatter.format_compilation_error(ce) for ce in compilation_errors
            ]

            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code="project_compilation_error",
                        message=(
                            "Project compilation failed outside the generated test class."
                        ),
                        details={
                            "tool": tool_name,
                            "tool_call_id": tool_call["id"],
                            "files_with_errors": sorted(files_with_errors),
                            "error_details": error_details,
                        },
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        if not has_error_for_target:
            qualified_class_name = (
                f"{state.package}.{state.class_name}"
                if state.package
                else state.class_name
            )
            method_signature = state.method_signature

            execution_issues: List[ExecutionIssue] = JavaMavenExecution(
                project_root, module_root=module_root
            ).get_execution_errors(qualified_class_name, method_signature)

            has_exec_error = len(execution_issues) > 0
            execution_failures: List[ExecutionIssue] = [
                ei for ei in execution_issues if ei.kind == "failure"
            ]
            execution_errors: List[ExecutionIssue] = [
                ei for ei in execution_issues if ei.kind == "error"
            ]

            result_payload["execution"] = {
                "status": "success" if not has_exec_error else "execution_error",
                "num_failures": len(execution_failures),
                "num_errors": len(execution_errors),
                "execution_failures": [
                    ErrorFormatter.format_execution_issue(ei)
                    for ei in execution_failures
                ],
                "execution_errors": [
                    ErrorFormatter.format_execution_issue(ei) for ei in execution_errors
                ],
            }
        else:
            result_payload["execution"] = {
                "status": "compilation_errors",
                "message": "Fix compilation errors in target test class before execution.",
            }

        outputs.append(
            ToolMessage(
                content=format_tool_ok(result_payload),
                tool_call_id=tool_call["id"],
            )
        )
