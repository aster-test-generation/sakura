from __future__ import annotations

from nltest.utils.compilation.maven import CompilationError
from nltest.utils.execution.maven import ExecutionIssue


class ErrorFormatter:
    """
    Shared utility for formatting compilation errors and execution issues
    into human-readable strings for tool output messages.
    """

    @staticmethod
    def format_compilation_error(compilation_error: CompilationError) -> str:
        """Format a compilation error into a readable string."""
        line = compilation_error.line if compilation_error.line is not None else "Unknown"

        if compilation_error.details:
            details_text = "\n".join(compilation_error.details)
        else:
            details_text = "No compiler details were provided."

        return (
            f"Line: {line}\n"
            f"Error Message: {compilation_error.message}\n"
            f"Error Details:\n{details_text}"
        )

    @staticmethod
    def format_execution_issue(execution_issue: ExecutionIssue) -> str:
        """Format an execution issue into a readable string."""
        class_name = execution_issue.class_name
        test_name = execution_issue.test_name
        test_case = f"{class_name}.{test_name}" if class_name and test_name else None

        line = execution_issue.line if execution_issue.line is not None else "Unknown"

        issue_type = execution_issue.kind or "Unknown"
        message = execution_issue.message or "No execution message was provided."
        stack_trace = execution_issue.stack_trace.strip() if execution_issue.stack_trace else "No stack trace was captured."

        return (
            f"Test Case: {test_case}\n"
            f"Issue Kind: {issue_type}\n"
            f"Message: {message}\n"
            f"Line: {line}\n"
            f"Stack Trace:\n{stack_trace}"
        )
