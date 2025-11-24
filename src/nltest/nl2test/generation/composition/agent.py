from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage, ToolCall
from langchain_core.tools import BaseTool

from nltest.nl2test.core.react_agent import ReActAgent
from nltest.nl2test.models import AgentState, DecompositionMode
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.utils.analysis.common_analysis import CommonAnalysis
from nltest.utils.compilation.maven import JavaMavenCompilation, CompilationError
from nltest.utils.execution.maven import ExecutionIssue, JavaMavenExecution
from nltest.utils.llm import LLMClient, FormatValidator
from nltest.utils.file_io.test_file_manager import TestFileManager, TestFileInfo
from nltest.utils.exceptions import ProjectCompilationError
from nltest.utils.pretty import RichLog
from nltest.utils.tool_messages import format_tool_error, format_tool_ok


class CompositionReActAgent(ReActAgent):
    def __init__(
            self,
            *,
            llm: LLMClient,
            tools: List[BaseTool],
            allow_duplicate_tools: List[BaseTool] | None = None,
            system_message: str,
            project_root: Path,
            max_iters: int = 30,
            parallelizable: bool = True,
    ):
        # Allow caller to inject system_message for parity with localization

        super().__init__(
            llm=llm,
            tools=tools,
            allow_duplicate_tools=allow_duplicate_tools,
            system_message=system_message,
            allow_parallelize=parallelizable,
            max_iters=max_iters,
        )
        self.project_root = Path(project_root)

    @staticmethod
    def _cleanup_empty_dirs(base_dir: Path, start_dir: Path) -> None:
        try:
            base = base_dir.resolve()
            current = start_dir.resolve()
        except Exception:
            return

        while current != base and base in current.parents:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    def prepare_tool_args(
            self, tool_name: str, raw_args: Dict[str, Any], _state: AgentState
    ) -> Tuple[str, Dict[str, Any]]:
        updated_args: Dict[str, Any] = raw_args

        # Normalize class name for CLDK inner classes
        normalize_class = {"get_method_details", "get_class_fields", "get_class_imports",
                           "get_class_constructors_and_factories", "get_getters_and_setters", "extract_method_code",
                           "get_call_site_details"}
        if tool_name in normalize_class:
            qualified_class_name = raw_args.get("qualified_class_name")
            normalized_class = CommonAnalysis.get_cldk_class_name(qualified_class_name)
            if normalized_class != qualified_class_name:
                # Clone only if diff
                if updated_args is raw_args:
                    updated_args = dict(updated_args)
                updated_args["qualified_class_name"] = normalized_class

        normalize_method_sig = {"get_call_site_details", "get_method_details", "extract_method_code"}
        if tool_name in normalize_method_sig:
            qualified_class_name = updated_args.get("qualified_class_name")
            method_signature = updated_args.get("method_signature")
            normalized_sig = CommonAnalysis.get_cldk_method_sig(
                qualified_class_name, method_signature
            )
            if normalized_sig != method_signature:
                if updated_args is raw_args:
                    updated_args = dict(updated_args)
                updated_args["method_signature"] = normalized_sig

        return tool_name, updated_args

    def process_tool_output(
            self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        """Process tool output and mutate state when appropriate."""
        tool_name = tool_call.get("name")
        handler_map = {
            "generate_test_code": self._process_generate_test_code_tool,
            "view_test_code": self._process_view_test_code_output,
            "compile_and_execute_test": self._process_compile_and_execute_test_output,
            "finalize": self._process_finalize_tool_output,
            "modify_scenario_comment": self._process_modify_scenario_comment_output,
        }
        handler = handler_map.get(tool_name, self._process_generic_tool_output)

        try:
            handler(tool_call, result, state, outputs)
        except ProjectCompilationError:
            raise
        except Exception as exc:
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code=type(exc).__name__,
                        message=str(exc),
                        details={
                            "tool": tool_name,
                            "tool_call_id": tool_call["id"],
                        },
                    ),
                    tool_call_id=tool_call["id"],
                )
            )

    # Subclass hook
    def process_llm_output(self, tool_call: ToolCall, state: AgentState) -> None:
        tool_name = tool_call.get("name")
        handler_map = {
            # "generate_test_code": self._process_generate_test_code_llm,
        }
        handler = handler_map.get(tool_name)

        # Only process if handler exists
        if handler:
            try:
                handler(tool_call, state)
            except Exception as exc:
                RichLog.error(f"Error on altering LLM output for {tool_name}: {exc}")

    # DEPRECATED
    def _process_generate_test_code_llm(self, tool_call: ToolCall, state: AgentState) -> None:
        placeholder = "(test_code redacted for token reduction)"
        raw_args = tool_call.get("args")
        parsed_args = self.llm.parse_tool_args(raw_args)

        if isinstance(parsed_args, dict):
            sanitized_args = dict(parsed_args)
        elif isinstance(raw_args, dict):
            sanitized_args = dict(parsed_args)
        else:
            sanitized_args = {}

        sanitized_args["test_code"] = placeholder

        try:
            if isinstance(raw_args, str):
                tool_call["args"] = json.dumps(sanitized_args, ensure_ascii=True, sort_keys=True)
            else:
                tool_call["args"] = sanitized_args
        except Exception:
            tool_call["args"] = sanitized_args

        call_id = tool_call.get("id")
        if not call_id:
            return

        for message in reversed(state.messages):
            if not isinstance(message, AIMessage):
                continue

            tool_calls = message.tool_calls
            for idx, call in enumerate(tool_calls):
                if call.get("id") != call_id:
                    continue
                tool_calls[idx] = tool_call

                # Raw provider metadata
                additional_kwargs = message.additional_kwargs
                if isinstance(additional_kwargs, dict):
                    kw_calls = additional_kwargs.get("tool_calls")
                    if isinstance(kw_calls, list) and idx < len(kw_calls):
                        kw_calls[idx] = tool_call
                return

    def _clear_previous_test_code_results(self, state: AgentState) -> None:
        """Redact stale tool outputs whenever new test code is generated."""
        # Note: Used for view_test_code and compile_and_execute_test, which would have outdated test code data
        call_lookup: Dict[str, str] = {}
        redacted = "(redacted since new code generated)"

        def _load_payload(content: Any) -> Tuple[Optional[Dict[str, Any]], str]:
            if isinstance(content, str):
                try:
                    return json.loads(content), "str"
                except json.JSONDecodeError:
                    return None, "str"
            if isinstance(content, dict):
                return content, "dict"
            return None, "other"

        for message in state.messages:
            # ToolMessage only optionally has tool name so backtrack to the Toolcall to get the name just in acse
            if isinstance(message, AIMessage):
                tool_calls = message.tool_calls or []
                for tc in tool_calls:
                    tc_id = tc.get("id")
                    tc_name = tc.get("name")
                    if tc_id and tc_name:
                        call_lookup[tc_id] = tc_name
                continue

            if not isinstance(message, ToolMessage):
                continue

            # message should only be ToolMessage
            tool_call_id = message.tool_call_id
            tool_name = message.name
            if not tool_name and tool_call_id:
                tool_name = call_lookup.get(tool_call_id)

            if tool_name not in {"view_test_code", "compile_and_execute_test"}:
                continue

            payload, payload_type = _load_payload(message.content)
            if not isinstance(payload, dict):
                continue

            # Skip errors
            if payload.get("status") != "ok":
                continue

            data = payload.get("data")
            if not isinstance(data, dict):
                continue

            updated = False
            if tool_name == "view_test_code":
                if "source" in data:
                    data["source"] = redacted
                    updated = True
            elif tool_name == "compile_and_execute_test":
                if "compilation" in data:
                    data["compilation"] = redacted
                    updated = True
                if "execution" in data:
                    data["execution"] = redacted
                    updated = True

            if not updated:
                continue

            # Type safe but should always be a str
            if payload_type == "str":
                message.content = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
            else:
                message.content = payload

    def _clear_previous_test_code_input(self, state: AgentState) -> None:
        """Redact older generate_test_code inputs when new code is saved."""
        # Call on generate_test_code tool calls
        redacted = "(redacted since new code generated)"
        latest_call_id: Optional[str] = None

        # Identify the most recent generate_test_code call id
        for message in reversed(state.messages):
            if not isinstance(message, AIMessage):
                continue
            tool_calls = message.tool_calls or []
            for call in reversed(tool_calls):
                if call.get("name") == "generate_test_code":
                    latest_call_id = call.get("id")
                    break
            if latest_call_id:
                break

        if not latest_call_id:
            return

        for message in state.messages:
            if not isinstance(message, AIMessage):
                continue

            tool_calls = message.tool_calls or []
            for idx, call in enumerate(tool_calls):
                if call.get("name") != "generate_test_code":
                    continue
                if call.get("id") == latest_call_id:
                    continue

                raw_args = call.get("args")
                parsed_args = self.llm.parse_tool_args(raw_args)

                if isinstance(parsed_args, dict):
                    sanitized_args = dict(parsed_args)
                elif isinstance(raw_args, dict):
                    sanitized_args = dict(raw_args)
                else:
                    sanitized_args = {}

                sanitized_args["test_code"] = redacted

                try:
                    if isinstance(raw_args, str):
                        call["args"] = json.dumps(sanitized_args, ensure_ascii=True, sort_keys=True)
                    else:
                        call["args"] = sanitized_args
                except Exception:
                    call["args"] = sanitized_args

                tool_calls[idx] = call

                additional_kwargs = message.additional_kwargs
                if isinstance(additional_kwargs, dict):
                    kw_calls = additional_kwargs.get("tool_calls")
                    if isinstance(kw_calls, list) and idx < len(kw_calls):
                        kw_calls[idx] = call

            message.tool_calls = tool_calls

    def _process_generate_test_code_tool(
            self, tool_call: ToolCall, result: Dict[str, Any], state: AgentState, outputs: List
    ) -> None:
        test_code = result.get("test_code")
        if isinstance(test_code, str):
            test_code = FormatValidator.sanitize_code_block(test_code)
        qualified_class_name = result.get("qualified_class_name")
        method_signature = result.get("method_signature")

        fm = TestFileManager(self.project_root)

        old_info: Optional[TestFileInfo] = None
        old_path: Optional[Path] = None
        if state.package and state.class_name:
            old_qcn = f"{state.package}.{state.class_name}" if state.package else state.class_name
            old_info = TestFileInfo(qualified_class_name=old_qcn)
            old_path = fm.target_path(old_info, encode_class_name=False)

        new_info = TestFileInfo(qualified_class_name=qualified_class_name, test_code=test_code)
        target_path = fm.target_path(new_info, encode_class_name=False)

        if old_path is not None and old_path == target_path:
            saved_qcn, saved_path = fm.save_single(
                new_info, encode_class_name=False, allow_overwrite=True
            )
        else:
            if old_info is not None and old_path is not None:
                fm.delete_single(
                    old_info,
                    encode_class_name=False,
                    strict=True,
                    max_attempts=2,
                    retry_delay=0.05,
                )
                self._cleanup_empty_dirs(fm.test_base_dir, old_path.parent)

            saved_qcn, saved_path = fm.save_single(new_info, encode_class_name=False)

        self._clear_previous_test_code_results(state)
        self._clear_previous_test_code_input(state)

        outputs.append(
            ToolMessage(
                content=format_tool_ok(
                    {
                        "message": "Saved test code.",
                        "qualified_class_name": saved_qcn,
                        "path": str(saved_path),
                    }
                ),
                tool_call_id=tool_call["id"],
            )
        )

        if "." in saved_qcn:
            pkg, simple_cls = saved_qcn.rsplit(".", 1)
            state.package = pkg or None
            state.class_name = simple_cls
        else:
            state.package = None
            state.class_name = saved_qcn
        state.method_signature = method_signature.strip() if method_signature.strip() else None

    def _process_view_test_code_output(
            self, tool_call: ToolCall, result: Dict[str, Any], state: AgentState, outputs: List
    ) -> None:
        tool_name = tool_call["name"]
        if not state.class_name:
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

        start_line = result.get("start_line")
        end_line = result.get("end_line")

        qcn = f"{state.package}.{state.class_name}" if state.package else state.class_name
        fm = TestFileManager(self.project_root)
        info = TestFileInfo(qualified_class_name=qcn)
        try:
            raw_code = fm.load(info, encode_class_name=False)
        except FileNotFoundError:
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code="test_file_not_found",
                        message=f"Test file not found for {qcn}.",
                        details={
                            "tool": tool_name,
                            "tool_call_id": tool_call["id"],
                            "qualified_class_name": qcn,
                        },
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        code_lines = raw_code.splitlines()
        total_lines = len(code_lines)
        if start_line > total_lines:
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code="invalid_line_range",
                        message="start_line is beyond the end of the file.",
                        details={
                            "tool": tool_name,
                            "tool_call_id": tool_call["id"],
                            "start_line": start_line,
                            "total_lines": total_lines,
                        },
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        applied_end_line = min(end_line, total_lines)
        sliced_source = "\n".join(
            code_lines[start_line - 1:applied_end_line]
        )
        payload = {
            "qualified_class_name": qcn,
            "source": sliced_source,
            "total_lines": total_lines,
            "start_line": start_line,
            "end_line": applied_end_line,
        }

        outputs.append(
            ToolMessage(
                content=format_tool_ok(payload),
                tool_call_id=tool_call["id"],
            )
        )

    def _format_compilation_error(self, compilation_error: CompilationError) -> str:
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

    def _format_execution_issue(self, execution_issue: ExecutionIssue) -> str:
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

    def _process_compile_and_execute_test_output(
            self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
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

        compilation_errors: List[CompilationError] = JavaMavenCompilation(self.project_root).get_compilation_errors()
        file_key = f"{state.class_name}.java"
        has_error_for_project = len(compilation_errors) > 0
        has_error_for_target = any(ef.file.endswith(file_key) for ef in compilation_errors)

        target_errors: List[str] = [self._format_compilation_error(compilation_error) for compilation_error in
                                    compilation_errors if compilation_error.file.endswith(file_key)]

        result_payload: Dict[str, Any] = {
            "compilation": {
                "status": "success" if not has_error_for_project else "compilation_error",
                "target_class_file": file_key,
                "has_errors_for_project": has_error_for_project,
                "has_errors_for_target": has_error_for_target,
                "error_details_for_target_class": target_errors,
            }
        }

        if has_error_for_project and not has_error_for_target:
            files_with_errors = [compilation_error.file for compilation_error in compilation_errors]
            error_details = [self._format_compilation_error(compilation_error) for compilation_error in
                             compilation_errors]
            raise ProjectCompilationError(
                "Unrelated project files failed to compile.",
                extra_info={
                    "files_with_errors": sorted(files_with_errors),
                    "error_details": error_details
                },
            )

        if not has_error_for_target:
            qualified_class_name = f"{state.package}.{state.class_name}" if state.package else state.class_name
            method_signature = state.method_signature
            execution_issues: List[ExecutionIssue] = JavaMavenExecution(self.project_root).get_execution_errors(
                qualified_class_name,
                method_signature)
            has_exec_error = len(execution_issues) > 0
            execution_failures: List[ExecutionIssue] = [execution_issue for execution_issue in execution_issues if
                                                        execution_issue.kind == "failure"]
            execution_errors: List[ExecutionIssue] = [execution_issue for execution_issue in execution_issues if
                                                      execution_issue.kind == "error"]

            result_payload["execution"] = {
                "status": "success" if not has_exec_error else "execution_error",
                "num_failures": len(execution_failures),
                "num_errors": len(execution_errors),
                "execution_failures": [self._format_execution_issue(execution_issue) for execution_issue in
                                       execution_failures],
                "execution_errors": [self._format_execution_issue(execution_issue) for execution_issue in
                                     execution_errors],
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

    def _process_finalize_tool_output(
            self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        comments = result
        state.final_comments = str(comments)
        state.finalize_called = True
        state.force_end_attempts = 0
        outputs.append(
            ToolMessage(
                content=format_tool_ok({"comments": str(comments)}),
                tool_call_id=tool_call["id"],
            )
        )
        setattr(self, "_end_now", True)

    def _process_modify_scenario_comment_output(
            self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        tool_name = tool_call["name"]
        step_id: Optional[int] = None
        comment: Optional[str] = None
        order: Optional[int] = None
        note: Optional[str] = None

        if isinstance(result, (list, tuple)) and len(result) == 2:
            if state.atomic_blocks is not None:
                order, note = int(result[0]), str(result[1])
            else:
                step_id, comment = int(result[0]), str(result[1])
        elif isinstance(result, dict):
            step_id = result.get("id")
            comment = result.get("comment")
            order = result.get("order")
            note = result.get("note")

        if state.atomic_blocks is not None:
            if order is None or note is None:
                outputs.append(
                    ToolMessage(
                        content=format_tool_error(
                            code="invalid_arguments",
                            message="Invalid modify_scenario_comment args for GRAMMATICAL mode.",
                            details={
                                "tool": tool_name,
                                "tool_call_id": tool_call["id"],
                                "order": order,
                            },
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )
                return

            found = False
            for block in state.atomic_blocks.atomic_blocks:
                if block.order == int(order):
                    block.notes = str(note)
                    found = True
                    break

            if found:
                outputs.append(
                    ToolMessage(
                        content=format_tool_ok(
                            {"message": f"Updated note for block order {order}.", "order": order}
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )
            else:
                outputs.append(
                    ToolMessage(
                        content=format_tool_error(
                            code="block_not_found",
                            message=f"No atomic block with order {order} exists.",
                            details={
                                "tool": tool_name,
                                "tool_call_id": tool_call["id"],
                                "order": order,
                            },
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )
            return

        if state.localized_scenario is not None:
            if step_id is None or comment is None:
                outputs.append(
                    ToolMessage(
                        content=format_tool_error(
                            code="invalid_arguments",
                            message="Invalid modify_scenario_comment args for GHERKIN mode.",
                            details={
                                "tool": tool_name,
                                "tool_call_id": tool_call["id"],
                                "step_id": step_id,
                            },
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )
                return

            found = False
            for entry in state.localized_scenario.setup:
                if entry.id == int(step_id):
                    entry.comments = str(comment)
                    found = True
                    break

            if not found:
                for grouped in state.localized_scenario.gherkin_groups:
                    for cluster in (
                        grouped.given,
                        grouped.when,
                        grouped.then,
                    ):
                        for entry in cluster:
                            if entry.id == int(step_id):
                                entry.comments = str(comment)
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

            if not found:
                for entry in state.localized_scenario.teardown:
                    if entry.id == int(step_id):
                        entry.comments = str(comment)
                        found = True
                        break

            if found:
                outputs.append(
                    ToolMessage(
                        content=format_tool_ok(
                            {"message": f"Updated comment for step id {step_id}.", "step_id": step_id}
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )
            else:
                outputs.append(
                    ToolMessage(
                        content=format_tool_error(
                            code="step_not_found",
                            message=f"No step with id {step_id} exists.",
                            details={
                                "tool": tool_name,
                                "tool_call_id": tool_call["id"],
                                "step_id": step_id,
                            },
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )
            return

        outputs.append(
            ToolMessage(
                content=format_tool_error(
                    code="state_unavailable",
                    message="No scenario or atomic blocks present to modify.",
                    details={
                        "tool": tool_name,
                        "tool_call_id": tool_call["id"],
                    },
                ),
                tool_call_id=tool_call["id"],
            )
        )

    def _process_generic_tool_output(
            self, tool_call: ToolCall, result: Any, _: AgentState, outputs: List
    ) -> None:
        outputs.append(
            ToolMessage(
                content=format_tool_ok(result),
                tool_call_id=tool_call["id"],
            )
        )
