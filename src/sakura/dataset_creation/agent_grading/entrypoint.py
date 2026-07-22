"""In-container entrypoint for the description-grading agent.

One container grades exactly one description: it clones the project from a
read-only local mirror mounted at ``/repo-src``, hard-resets to the pinned
commit, runs the grading agent, and writes one result JSON under ``/out`` for
the host to aggregate.

Runs as ``python -m agent_grading.entrypoint --task-spec /out/...`` inside the
image (where this package is copied as top-level ``agent_grading``); every
import is relative so the module is location-independent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any

from .agent import AgentResult, GraderAgentConfig, GradingAgent
from .git_reset import reset_to_commit
from .prompt_renderer import PromptRenderer
from .schema import validate_structured_grades

TASK_PATH = Path("/work/task.json")
REPO_PATH = Path("/work/repo")
OUTPUT_ROOT = Path("/out")
REPO_SOURCE = Path("/repo-src")


class ContainerGradingError(RuntimeError):
    """Raised when the in-container grading run cannot continue."""


def _load_spec(task_path: Path) -> dict[str, Any]:
    if not task_path.exists():
        raise ContainerGradingError(f"Task spec does not exist: {task_path}")
    with task_path.open(encoding="utf-8") as f:
        return json.load(f)


def _clone_local_repo(source: Path, repo_path: Path) -> None:
    if repo_path.exists():
        shutil.rmtree(repo_path)
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", str(source), str(repo_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ContainerGradingError(
            f"Failed to clone {source} into {repo_path}:\n{detail}"
        )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def _result_fields(result: AgentResult) -> dict[str, Any]:
    return {
        "agent_text": result.text,
        "usage": result.usage,
        "model_usage": result.model_usage,
        "total_cost_usd": result.total_cost_usd,
        "num_turns": result.num_turns,
        "duration_ms": result.duration_ms,
        "duration_api_ms": result.duration_api_ms,
        "stop_reason": result.stop_reason,
    }


async def run_async(task_path: Path = TASK_PATH) -> None:
    spec = _load_spec(task_path)
    entry = spec["entry"]
    output_root = Path(spec.get("output_root", str(OUTPUT_ROOT)))
    result_path = output_root / spec["result_relpath"]
    repo_source = Path(spec.get("repo_source", str(REPO_SOURCE)))
    pinned_commit = spec["pinned_commit"]

    record: dict[str, Any] = {
        "schema_version": 1,
        "id": entry["id"],
        "status": "error",
        "error": None,
        "grades": None,
        "rationales": None,
        "structured_raw": None,
        "model": spec["agent"]["model"],
        "pinned_commit": pinned_commit,
        "file_path": spec.get("file_path"),
    }

    print(
        f"[container] grading id={entry['id']} "
        f"{entry['qualified_class_name']}::{entry['method_signature']}",
        flush=True,
    )
    try:
        _clone_local_repo(repo_source, REPO_PATH)
        reset_to_commit(REPO_PATH, pinned_commit)

        agent = GradingAgent(
            GraderAgentConfig(
                model=spec["agent"]["model"],
                thinking=spec["agent"].get("thinking"),
            ),
            PromptRenderer(),
        )
        context = {
            "entry": entry,
            "project_description": spec["project_description"],
            "file_path": spec.get("file_path"),
            "criteria": spec["criteria"],
        }
        result = await agent.run(
            cwd=REPO_PATH, system_context=context, user_context=context
        )

        record.update(_result_fields(result))
        record["structured_raw"] = result.structured
        grades = validate_structured_grades(result.structured)
        record["grades"] = grades
        record["rationales"] = {
            "fidelity": result.structured["fidelity_rationale"],
            "perceived_level": result.structured["perceived_rationale"],
        }
        record["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - persist task-level failures
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["error_traceback"] = traceback.format_exc()

    _write_json_atomic(result_path, record)
    print(f"[container] {record['status']}", flush=True)


def run(task_path: Path = TASK_PATH) -> None:
    asyncio.run(run_async(task_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Internal Docker entrypoint for agent description grading."
    )
    parser.add_argument(
        "--task-spec",
        type=Path,
        default=TASK_PATH,
        help="Path to the mounted JSON task spec. Defaults to /work/task.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(task_path=args.task_spec)


if __name__ == "__main__":
    main()
