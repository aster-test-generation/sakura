from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, ToolMessage, ToolCall
from langchain_core.tools import BaseTool

from nltest.nl2test.core.react_agent import ReActAgent
from nltest.nl2test.core.message_redactor import MessageRedactor
from nltest.nl2test.generation.common.cldk_normalizer import CLDKArgNormalizer
from nltest.nl2test.generation.common.compilation_execution import (
    CompilationExecutionMixin,
)
from nltest.nl2test.models import AgentState
from nltest.utils.llm import LLMClient, FormatValidator
from nltest.utils.constants import TEST_DIR
from nltest.utils.file_io.test_file_manager import TestFileManager, TestFileInfo
from nltest.utils.exceptions import ProjectCompilationError
from nltest.utils.tool_messages import format_tool_error, format_tool_ok


class CompositionReActAgent(ReActAgent, CompilationExecutionMixin):
    """
    Composition agent that generates executable Java tests from localized scenarios.

    Takes localized step sequences and generates compilable Java test code,
    managing the test file lifecycle and validating compilation/execution.
    """

    # Redaction placeholder for token reduction
    _CODE_REDACTED = "(redacted since new code generated)"

    _JAVA_PACKAGE_DECL_RE = re.compile(
        r"(?m)^\s*package\s+(?P<package>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*;"
    )
    _JAVA_TOP_LEVEL_TYPE_RE = re.compile(
        r"(?m)^(?:\s*@\w+(?:\([^)]*\))?\s*)*(?:public\s+)?"
        r"(?:abstract\s+|final\s+)?(?:class|interface|enum|record)\s+"
        r"(?P<name>[A-Za-z_]\w*)"
    )

    @staticmethod
    def _extract_package_from_code(code: str) -> str | None:
        match = CompositionReActAgent._JAVA_PACKAGE_DECL_RE.search(code)
        return match.group("package") if match else None

    @staticmethod
    def _extract_top_level_type_name(code: str) -> str | None:
        match = CompositionReActAgent._JAVA_TOP_LEVEL_TYPE_RE.search(code)
        return match.group("name") if match else None

    @staticmethod
    def _normalize_test_qualified_class_name(test_code: str, qcn: str) -> str:
        declared_package = (
            CompositionReActAgent._extract_package_from_code(test_code) or ""
        )

        provided_package, sep, provided_class_name = qcn.rpartition(".")
        if not sep:
            provided_package = ""
            provided_class_name = qcn

        if not provided_class_name:
            provided_class_name = (
                CompositionReActAgent._extract_top_level_type_name(test_code) or qcn
            )

        final_package = provided_package or declared_package
        if final_package:
            return f"{final_package}.{provided_class_name}"
        return provided_class_name

    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: List[BaseTool],
        allow_duplicate_tools: List[BaseTool] | None = None,
        system_message: str,
        project_root: Path,
        test_base_dir: str | Path | None = None,
        module_root: Path | None = None,
        max_iters: int = 30,
        parallelizable: bool = True,
    ):
        super().__init__(
            llm=llm,
            tools=tools,
            allow_duplicate_tools=allow_duplicate_tools,
            system_message=system_message,
            allow_parallelize=parallelizable,
            max_iters=max_iters,
        )
        self.project_root = Path(project_root)
        self.test_base_dir = test_base_dir
        resolved_module_root = (
            Path(module_root).expanduser() if module_root is not None else None
        )
        if resolved_module_root is not None and not resolved_module_root.is_absolute():
            resolved_module_root = self.project_root / resolved_module_root
        self.module_root = (
            resolved_module_root.resolve() if resolved_module_root is not None else None
        )

    @staticmethod
    def _cleanup_empty_dirs(base_dir: Path, start_dir: Path) -> None:
        """Remove empty directories up to base_dir after file deletion."""
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

    def _get_test_file_manager(self) -> TestFileManager:
        if self.project_root is None:
            raise ValueError("project_root is not configured")
        test_base_dir = self.test_base_dir or TEST_DIR
        return TestFileManager(self.project_root, test_base_dir=test_base_dir)

    def prepare_tool_args(
        self, tool_name: str, raw_args: Dict[str, Any], state: AgentState
    ) -> Tuple[str, Dict[str, Any]]:
        """Normalize tool arguments for CLDK compatibility."""
        _ = state
        updated_args = CLDKArgNormalizer.normalize_args(tool_name, raw_args)
        return tool_name, updated_args

    def process_tool_output(
        self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        """Process tool output and mutate state when appropriate."""
        if self._append_tool_error_if_needed(tool_call, result, outputs):
            return

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

    def process_llm_output(self, tool_call: ToolCall, state: AgentState) -> None:
        """Hook for processing LLM outputs (currently unused)."""
        tool_name = tool_call.get("name")
        handler_map = {
            # "generate_test_code": self._process_generate_test_code_llm,  # DEPRECATED
        }
        handler = handler_map.get(tool_name)

        if handler:
            try:
                handler(tool_call, state)
            except Exception:
                pass  # Silently ignore LLM output processing errors

    def _redact_previous_test_code_results(self, state: AgentState) -> None:
        """Redact stale tool outputs whenever new test code is generated."""
        MessageRedactor.redact_tool_outputs_by_tool(
            state,
            tool_redactions={
                "view_test_code": {"source": self._CODE_REDACTED},
                "compile_and_execute_test": {
                    "compilation": self._CODE_REDACTED,
                    "execution": self._CODE_REDACTED,
                },
            },
        )

    def _redact_previous_test_code_inputs(self, state: AgentState) -> None:
        """Redact older generate_test_code inputs when new code is saved."""
        MessageRedactor.redact_tool_inputs(
            state,
            tool_name="generate_test_code",
            keys_to_redact={"test_code": self._CODE_REDACTED},
            llm_client=self.llm,
            exclude_latest=True,
        )

    def _process_generate_test_code_tool(
        self,
        tool_call: ToolCall,
        result: Dict[str, Any],
        state: AgentState,
        outputs: List,
    ) -> None:
        test_code = result.get("test_code")
        if isinstance(test_code, str):
            test_code = FormatValidator.sanitize_code_block(test_code)
        qualified_class_name = result.get("qualified_class_name")
        method_signature = result.get("method_signature")

        if not isinstance(test_code, str) or not test_code.strip():
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code="invalid_test_code",
                        message="generate_test_code must provide non-empty Java source in test_code.",
                        details={
                            "tool": "generate_test_code",
                            "tool_call_id": tool_call["id"],
                        },
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        if (
            not isinstance(qualified_class_name, str)
            or not qualified_class_name.strip()
        ):
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code="invalid_qualified_class_name",
                        message=(
                            "generate_test_code must provide a fully qualified test class name in qualified_class_name."
                        ),
                        details={
                            "tool": "generate_test_code",
                            "tool_call_id": tool_call["id"],
                        },
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        if not isinstance(method_signature, str) or not method_signature.strip():
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code="invalid_method_signature",
                        message=(
                            "generate_test_code must provide the generated @Test method signature in method_signature."
                        ),
                        details={
                            "tool": "generate_test_code",
                            "tool_call_id": tool_call["id"],
                        },
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        requested_qcn = qualified_class_name.strip()
        normalized_qcn = self._normalize_test_qualified_class_name(
            test_code, requested_qcn
        )
        qualified_class_name = normalized_qcn
        method_signature = method_signature.strip()

        fm = self._get_test_file_manager()

        old_info: Optional[TestFileInfo] = None
        old_path: Optional[Path] = None
        if state.package and state.class_name:
            old_qcn = (
                f"{state.package}.{state.class_name}"
                if state.package
                else state.class_name
            )
            old_info = TestFileInfo(qualified_class_name=old_qcn)
            old_path = fm.target_path(old_info, encode_class_name=False)

        new_info = TestFileInfo(
            qualified_class_name=qualified_class_name, test_code=test_code
        )
        target_path = fm.target_path(new_info, encode_class_name=False)

        if old_path is not None and old_path == target_path:
            saved_qcn, saved_path = fm.save_single(
                new_info,
                encode_class_name=False,
                allow_overwrite=True,
                sync_names=True,
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

            saved_qcn, saved_path = fm.save_single(
                new_info,
                encode_class_name=False,
                sync_names=True,
            )

        self._redact_previous_test_code_results(state)
        self._redact_previous_test_code_inputs(state)

        payload: Dict[str, Any] = {
            "message": "Saved test code.",
            "qualified_class_name": saved_qcn,
            "path": str(saved_path),
        }
        if requested_qcn != normalized_qcn:
            payload["input_qualified_class_name"] = requested_qcn
            payload["normalized_qualified_class_name"] = normalized_qcn

        outputs.append(
            ToolMessage(
                content=format_tool_ok(payload),
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
        state.method_signature = method_signature

    def _process_view_test_code_output(
        self,
        tool_call: ToolCall,
        result: Dict[str, Any],
        state: AgentState,
        outputs: List,
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

        start_line_raw = result.get("start_line")
        end_line_raw = result.get("end_line")
        if not isinstance(start_line_raw, int) or not isinstance(end_line_raw, int):
            outputs.append(
                ToolMessage(
                    content=format_tool_error(
                        code="invalid_line_range",
                        message="start_line and end_line must be integers.",
                        details={
                            "tool": tool_name,
                            "tool_call_id": tool_call["id"],
                            "start_line": start_line_raw,
                            "end_line": end_line_raw,
                        },
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
            return

        start_line = start_line_raw
        end_line = end_line_raw

        qcn = (
            f"{state.package}.{state.class_name}" if state.package else state.class_name
        )
        fm = self._get_test_file_manager()
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
        sliced_source = "\n".join(code_lines[start_line - 1 : applied_end_line])
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

    def _process_compile_and_execute_test_output(
        self, tool_call: ToolCall, result: Any, state: AgentState, outputs: List
    ) -> None:
        """Process compile_and_execute_test using shared mixin."""
        self.process_compile_and_execute(tool_call, state, outputs)

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

        # Handle GRAMMATICAL mode (atomic_blocks)
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
                            {
                                "message": f"Updated note for block order {order}.",
                                "order": order,
                            }
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

        # Handle GHERKIN mode (localized_scenario)
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
            # Check setup steps
            for entry in state.localized_scenario.setup:
                if entry.id == int(step_id):
                    entry.comments = str(comment)
                    found = True
                    break

            # Check gherkin groups
            if not found:
                for grouped in state.localized_scenario.gherkin_groups:
                    for cluster in (grouped.given, grouped.when, grouped.then):
                        for entry in cluster:
                            if entry.id == int(step_id):
                                entry.comments = str(comment)
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

            # Check teardown steps
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
                            {
                                "message": f"Updated comment for step id {step_id}.",
                                "step_id": step_id,
                            }
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

        # No scenario or atomic blocks present
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
        self._append_tool_output(tool_call, result, outputs)
