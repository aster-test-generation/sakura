from __future__ import annotations
_A=None
from pathlib import Path
from typing import Any,Dict,List
from langchain_core.messages import ToolMessage,ToolCall
from sakura.nl2test.models import AgentState
from sakura.utils.compilation.maven import CompilationError,JavaMavenCompilation
from sakura.utils.execution.maven import ExecutionIssue,JavaMavenExecution
from sakura.utils.formatting import ErrorFormatter
from sakura.utils.tool_messages import format_tool_error,format_tool_ok
class CompilationExecutionMixin:
	'\n    Shared mixin providing compile_and_execute_test processing logic.\n\n    Requirements:\n    - The using class must have a `project_root` attribute (Path or None)\n    - Optional `module_root` attribute for multi-module builds\n    ';project_root:Path|_A;module_root:Path|_A=_A
	def process_compile_and_execute(G,tool_call:ToolCall,state:AgentState,outputs:List[ToolMessage])->_A:
		'\n        Process compile_and_execute_test tool output.\n\n        Runs Maven compilation, checks for errors in target class,\n        runs test execution if compilation succeeds, and builds result payload.\n\n        Args:\n            tool_call: The tool call being processed\n            state: Current agent state with class_name, package, method_signature\n            outputs: List to append ToolMessage results to\n\n        Notes:\n            If unrelated project files fail to compile, this tool reports a structured\n            error payload rather than raising an exception.\n        ';W='execution';V='success';P='status';O='/';N='tool_call_id';M='tool';E=outputs;C='id';B=tool_call;A=state;H=B['name']
		if not A.class_name or not A.method_signature:E.append(ToolMessage(content=format_tool_error(code='no_active_test',message='No test code has been generated or saved.',details={M:H,N:B[C]}),tool_call_id=B[C]));return
		if G.project_root is _A:E.append(ToolMessage(content=format_tool_error(code='no_project_root',message='Project root is not configured; cannot compile or execute tests.',details={M:H,N:B[C]}),tool_call_id=B[C]));return
		Q=G.project_root;R=G.module_root;D:List[CompilationError]=JavaMavenCompilation(Q,module_root=R).get_compilation_errors();F=f"{A.class_name}.java";X=f"{A.package.replace('.',O)}/{F}"if A.package else F
		def S(err:str)->bool:
			A=err.replace('\\',O)
			if O in A:return A.endswith(X)
			return A.endswith(F)
		I=len(D)>0;J=any(S(A.file)for A in D);Y:List[str]=[ErrorFormatter.format_compilation_error(A)for A in D if S(A.file)];K:Dict[str,Any]={'compilation':{P:V if not I else'compilation_error','target_class_file':F,'has_errors_for_project':I,'has_errors_for_target':J,'error_details_for_target_class':Y}}
		if I and not J:Z=[A.file for A in D];a=[ErrorFormatter.format_compilation_error(A)for A in D];E.append(ToolMessage(content=format_tool_error(code='project_compilation_error',message='Project compilation failed outside the generated test class.',details={M:H,N:B[C],'files_with_errors':sorted(Z),'error_details':a}),tool_call_id=B[C]));return
		if not J:b=f"{A.package}.{A.class_name}"if A.package else A.class_name;c=A.method_signature;L:List[ExecutionIssue]=JavaMavenExecution(Q,module_root=R).get_execution_errors(b,c);d=len(L)>0;T:List[ExecutionIssue]=[A for A in L if A.kind=='failure'];U:List[ExecutionIssue]=[A for A in L if A.kind=='error'];K[W]={P:V if not d else'execution_error','num_failures':len(T),'num_errors':len(U),'execution_failures':[ErrorFormatter.format_execution_issue(A)for A in T],'execution_errors':[ErrorFormatter.format_execution_issue(A)for A in U]}
		else:K[W]={P:'compilation_errors','message':'Fix compilation errors in target test class before execution.'}
		E.append(ToolMessage(content=format_tool_ok(K),tool_call_id=B[C]))