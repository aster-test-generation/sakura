from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from nltest.nl2test.core.react_agent import ReActAgent
from nltest.nl2test.model.models import AgentState, NL2TestInput
from nltest.utils.execution import JavaCompilation
from nltest.utils.execution.execution import JavaExecution
from nltest.utils.file_io import TestFileManager, TestFileInfo
from nltest.utils.llm.llm_client import LLMClient


class SupervisorTools:
    def __init__(
        self,
        *,
        llm: LLMClient,
        project_root: Path,
        nl2_input: NL2TestInput,
        localization_agent: Optional[ReActAgent] = None,
        composition_agent: Optional[ReActAgent] = None,
    ) -> None:
        self.llm = llm
        self.project_root = Path(project_root)
        self.nl2_input = nl2_input
        self.localization_agent = localization_agent
        self.composition_agent = composition_agent
        self._tasks: Dict[str, Dict[str, Any]] = {}

    class _EmptyArgs(BaseModel):
        pass

    class _GoalArgs(BaseModel):
        goal: str = Field(..., description="Overall goal (natural language).")
        nl_description: Optional[str] = Field(None, description="If omitted, uses the NL2TestInput description.")

    class _AssignArgs(BaseModel):
        instructions: str = Field(..., description="Actionable instruction for the delegated agent.")
        state_json: Optional[str] = Field(None, description="Optional JSON-encoded seed state (e.g., atomic_blocks).")

    class _TaskIdArgs(BaseModel):
        task_id: str = Field(..., description="Supervisor task id returned by assignment calls.")

    class _CollectArgs(BaseModel):
        task_ids: List[str] = Field(..., description="Task ids to collect/merge artifacts from.")

    def make_tools(self) -> List[BaseTool]:
        return [
            self._make_set_goal_tool(),
            self._make_assign_localization_tool(),
            self._make_assign_composition_tool(),
            self._make_get_status_tool(),
            self._make_list_tasks_tool(),
            self._make_collect_artifacts_tool(),
            self._make_compile_tests_tool(),
            self._make_execute_tests_tool(),
            self._make_finalize_tool(),
        ]

    def _make_set_goal_tool(self) -> BaseTool:
        def _set_goal(goal: str, nl_description: Optional[str] = None) -> Dict[str, Any]:
            tid = f"goal:{uuid.uuid4().hex[:8]}"
            self._tasks[tid] = {
                "id": tid, "agent": "supervisor", "type": "goal", "status": "set",
                "goal": goal, "nl_description": nl_description or self.nl2_input.description
            }
            return {"task_id": tid, "status": "set", "goal": goal}

        return StructuredTool.from_function(
            func=_set_goal,
            name="set_goal",
            description="Register the overall goal for subsequent sub-tasks.",
            args_schema=self._GoalArgs,
        )

    def _make_assign_localization_tool(self) -> BaseTool:
        def _assign_localization(instructions: str, state_json: Optional[str] = None) -> Dict[str, Any]:
            if not self.localization_agent:
                return {"error": "localization_agent is not configured"}
            init_state: AgentState = AgentState()
            if state_json:
                try:
                    payload = json.loads(state_json)
                    for k, v in payload.items():
                        setattr(init_state, k, v)
                except json.JSONDecodeError:
                    pass
            result_state: AgentState = self.localization_agent.invoke(instructions, init_state)
            tid = f"loc:{uuid.uuid4().hex[:8]}"
            self._tasks[tid] = {"id": tid, "agent": "localization", "status": "completed", "result": result_state}
            return {"task_id": tid, "status": "completed"}

        return StructuredTool.from_function(
            func=_assign_localization,
            name="assign_localization_task",
            description="Delegate a sub-goal to the Localization agent.",
            args_schema=self._AssignArgs,
        )

    def _make_assign_composition_tool(self) -> BaseTool:
        def _assign_composition(instructions: str, state_json: Optional[str] = None) -> Dict[str, Any]:
            if not self.composition_agent:
                return {"error": "composition_agent is not configured"}
            init_state: AgentState = AgentState()
            if state_json:
                try:
                    payload = json.loads(state_json)
                    for k, v in payload.items():
                        setattr(init_state, k, v)
                except json.JSONDecodeError:
                    pass
            result_state: AgentState = self.composition_agent.invoke(instructions, init_state)
            tid = f"comp:{uuid.uuid4().hex[:8]}"
            self._tasks[tid] = {"id": tid, "agent": "composition", "status": "completed", "result": result_state}
            return {"task_id": tid, "status": "completed"}

        return StructuredTool.from_function(
            func=_assign_composition,
            name="assign_composition_task",
            description="Delegate a sub-goal to the Composition agent.",
            args_schema=self._AssignArgs,
        )

    def _make_get_status_tool(self) -> BaseTool:
        def _get_status(task_id: str) -> Dict[str, Any]:
            task = self._tasks.get(task_id)
            if not task:
                return {"task_id": task_id, "status": "unknown"}
            return {"task_id": task_id, "status": task.get("status", "unknown"), "agent": task.get("agent")}

        return StructuredTool.from_function(
            func=_get_status,
            name="get_status",
            description="Get the status of a previously assigned task.",
            args_schema=self._TaskIdArgs,
        )

    def _make_list_tasks_tool(self) -> BaseTool:
        def _list_tasks() -> Dict[str, Any]:
            return {"tasks": [{"id": t["id"], "agent": t.get("agent"), "status": t.get("status")}
                              for t in self._tasks.values()]}

        return StructuredTool.from_function(
            func=_list_tasks,
            name="list_tasks",
            description="List all tasks known to the Supervisor.",
            args_schema=self._EmptyArgs,
        )

    def _make_collect_artifacts_tool(self) -> BaseTool:
        def _collect(task_ids: List[str]) -> Dict[str, Any]:
            merged: Dict[str, Any] = {}
            for tid in task_ids:
                if tid in self._tasks and "result" in self._tasks[tid]:
                    res = self._tasks[tid]["result"]
                    if hasattr(res, "model_dump"):
                        data = res.model_dump()
                    elif isinstance(res, dict):
                        data = res
                    else:
                        data = res.__dict__
                    merged.update(data)
            return {"merged_state": merged, "num_sources": len(task_ids)}

        return StructuredTool.from_function(
            func=_collect,
            name="collect_artifacts",
            description="Collect and shallow-merge result states from tasks.",
            args_schema=self._CollectArgs,
        )

    def _make_compile_tests_tool(self) -> BaseTool:
        def _compile_tests() -> Dict[str, Any]:
            info = TestFileInfo.from_nl2test_input(self.nl2_input)
            test_fqn = TestFileManager(self.project_root).make_test_fqn(info)
            compilation_feedback = JavaCompilation.compile(str(self.project_root), test_fqn)
            if isinstance(compilation_feedback, dict):
                return compilation_feedback
            return {"has_errors_for_target": False, "target_class_file": test_fqn, "errors_by_file": {}}

        return StructuredTool.from_function(
            func=_compile_tests,
            name="compile_test_suite",
            description="Compile the current generated test suite.",
            args_schema=self._EmptyArgs,
        )

    def _make_execute_tests_tool(self) -> BaseTool:
        def _execute_tests() -> Dict[str, Any]:
            info = TestFileInfo.from_nl2test_input(self.nl2_input)
            test_fqn = TestFileManager(self.project_root).make_test_fqn(info)
            return JavaExecution.execute(str(self.project_root), test_fqn)

        return StructuredTool.from_function(
            func=_execute_tests,
            name="run_test_suite",
            description="Execute the test suite and return feedback.",
            args_schema=self._EmptyArgs,
        )

    def _make_finalize_tool(self) -> BaseTool:
        def _finalize() -> Dict[str, Any]:
            return {"status": "finalized"}

        return StructuredTool.from_function(
            func=_finalize,
            name="finalize",
            description="Mark the goal as completed when compile + run are satisfactory.",
            args_schema=self._EmptyArgs,
        )
