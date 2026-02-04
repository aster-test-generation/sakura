from __future__ import annotations
_A='Unknown'
from sakura.utils.compilation.maven import CompilationError
from sakura.utils.execution.maven import ExecutionIssue
class ErrorFormatter:
	'\n    Shared utility for formatting compilation errors and execution issues\n    into human-readable strings for tool output messages.\n    '
	@staticmethod
	def format_compilation_error(compilation_error:CompilationError)->str:
		'Format a compilation error into a readable string.';A=compilation_error;C=A.line if A.line is not None else _A
		if A.details:B='\n'.join(A.details)
		else:B='No compiler details were provided.'
		return f"Line: {C}\nError Message: {A.message}\nError Details:\n{B}"
	@staticmethod
	def format_execution_issue(execution_issue:ExecutionIssue)->str:'Format an execution issue into a readable string.';A=execution_issue;B=A.class_name;C=A.test_name;D=f"{B}.{C}"if B and C else None;E=A.line if A.line is not None else _A;F=A.kind or _A;G=A.message or'No execution message was provided.';H=A.stack_trace.strip()if A.stack_trace else'No stack trace was captured.';return f"""Test Case: {D}
Issue Kind: {F}
Message: {G}
Line: {E}
Stack Trace:
{H}"""