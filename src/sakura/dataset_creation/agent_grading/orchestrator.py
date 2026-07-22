"""Host-side orchestration for agent description grading.

Derives the human grader order, prepares one git mirror per project, runs one
Docker container per description, and aggregates grades into
``outputs/descriptions_sample/agent_graded/<label>.json`` — a file with the
exact schema of a human reviewer's ``graded/<user>.json`` so downstream
comparison treats the agent as just another reviewer. Per-entry rationale,
usage, and error details land next to it under ``agent_graded/<label>/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sakura.dataset_creation.description_grading.grading_criteria import (
    criteria_payload,
)
from sakura.dataset_creation.description_grading.session import (
    Grades,
    load_grader_order,
    load_project_descriptions,
)
from sakura.test2nl.model.models import Test2NLEntry
from sakura.utils.file_io.structured_data_manager import StructuredDataManager
from sakura.utils.pretty.color_logger import RichLog

from .docker_runner import DockerSandboxError, DockerSandboxRunner
from .file_paths import derive_java_file_path
from .git_reset import resolve_reset_target
from .repo_cache import CACHE_DIR_NAME, ensure_repo_mirror

SCHEMA_VERSION = 3
DATASETS_DIR_NAME = "resources/datasets"


def build_task_spec(
    entry: Test2NLEntry,
    *,
    project_description: str,
    file_path: str | None,
    pinned_commit: str,
    model: str,
    thinking: str | None,
    result_relpath: str,
) -> dict[str, Any]:
    """The JSON spec one container consumes to grade one description.

    The true abstraction level is stripped so the agent stays as blind as a
    human reviewer (mirroring GradingSession.entry_payload).
    """
    return {
        "schema_version": 1,
        "entry": entry.model_dump(mode="json", exclude={"abstraction_level"}),
        "project_description": project_description,
        "file_path": file_path,
        "pinned_commit": pinned_commit,
        "result_relpath": result_relpath,
        "agent": {"model": model, "thinking": thinking},
        "criteria": criteria_payload(),
    }


class AggregateGradeFile:
    """The agent's ``<label>.json``, schema-compatible with human grade files."""

    def __init__(
        self,
        path: Path,
        label: str,
        ordered_ids: list[int],
        *,
        reset: bool = False,
    ) -> None:
        self.path = path
        self.label = label
        self.ordered_ids = list(ordered_ids)
        self.grades: dict[int, Grades] = {}
        self.archived_backup: Path | None = (
            self._archive_existing() if reset else None
        )
        self._load_existing()

    def _archive_existing(self) -> Path | None:
        if not self.path.exists():
            return None
        backup = self.path.with_suffix(".json.bak")
        backup.unlink(missing_ok=True)
        self.path.rename(backup)
        return backup

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid existing grade file: {self.path}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Existing grade file must be an object: {self.path}")
        if data.get("schema_version") != SCHEMA_VERSION or data.get("user") != self.label:
            raise ValueError(
                f"Incompatible existing grade file: {self.path}. "
                "Rerun with --reset to archive it and start over."
            )
        saved_entries = data.get("entries")
        if not isinstance(saved_entries, list) or not all(
            isinstance(entry, dict) for entry in saved_entries
        ):
            raise ValueError(f"Existing grade entries must be a list: {self.path}")
        saved_ids = [entry.get("id") for entry in saved_entries]
        if saved_ids != self.ordered_ids:
            raise ValueError(
                "Existing grade file does not match the current grader order: "
                f"{self.path}. Rerun with --reset to archive it and start over."
            )
        for saved in saved_entries:
            if saved.get("grades") is not None:
                self.grades[saved["id"]] = Grades.model_validate(saved["grades"])

    def grade_for(self, entry_id: int) -> Grades | None:
        return self.grades.get(entry_id)

    def set_grade(self, entry_id: int, grades: Grades) -> None:
        if entry_id not in set(self.ordered_ids):
            raise ValueError(f"Entry ID is not in the grader order: {entry_id}")
        self.grades[entry_id] = grades

    def save(self) -> None:
        document = {
            "schema_version": SCHEMA_VERSION,
            "user": self.label,
            "selection": {"kind": "full", "path": None},
            "entries": [
                {
                    "id": entry_id,
                    "grades": (
                        self.grades[entry_id].model_dump()
                        if entry_id in self.grades
                        else None
                    ),
                }
                for entry_id in self.ordered_ids
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        StructuredDataManager._atomic_write_text(
            self.path,
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        )


def _write_details(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    StructuredDataManager._atomic_write_text(
        path, json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    )


def _synthesized_error_record(
    entry: Test2NLEntry, model: str, error: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": entry.id,
        "status": "error",
        "error": error,
        "grades": None,
        "rationales": None,
        "structured_raw": None,
        "model": model,
    }


def run_agent_grading(
    *,
    num: int,
    model: str,
    label: str,
    thinking: str | None,
    repo_root: Path,
    runner: DockerSandboxRunner,
    reset: bool = False,
) -> dict[str, Any]:
    """Grade the first ``num`` entries of the human grader order with the agent.

    Setup problems (missing CSVs/checkouts/mirrors, incompatible aggregate)
    fail fast; per-entry problems (container failure, invalid structured
    output) are recorded in the details file and skipped.

    Returns:
        Summary dict with graded/errored/skipped counts and summed cost.
    """
    descriptions_dir = repo_root / "outputs" / "descriptions_sample"
    entries = load_grader_order(descriptions_dir)
    project_descriptions = load_project_descriptions(descriptions_dir)

    output_dir = descriptions_dir / "agent_graded"
    run_root = output_dir / label
    aggregate = AggregateGradeFile(
        output_dir / f"{label}.json",
        label,
        [entry.id for entry in entries],
        reset=reset,
    )
    if aggregate.archived_backup is not None:
        RichLog.info(f"Archived previous grades to {aggregate.archived_backup}")
    aggregate.save()

    if num > len(entries):
        RichLog.warn(
            f"--num {num} exceeds the {len(entries)} available entries; "
            "grading all of them."
        )
    targets = entries[:num]
    pending = [
        entry for entry in targets if aggregate.grade_for(entry.id) is None
    ]
    skipped = len(targets) - len(pending)
    if skipped:
        RichLog.info(f"Resuming: {skipped} of {len(targets)} entries already graded.")

    # One mirror + pinned commit per distinct project, prepared up front so a
    # broken checkout fails the run before any container spends tokens.
    checkouts: dict[str, Path] = {}
    pinned_commits: dict[str, str] = {}
    mirrors: dict[str, Path] = {}
    for project in sorted({entry.project_name for entry in pending}):
        checkout = repo_root / DATASETS_DIR_NAME / project
        if not checkout.is_dir():
            raise FileNotFoundError(
                f"Project checkout does not exist: {checkout}. "
                "Initialize the dataset submodules first."
            )
        checkouts[project] = checkout
        pinned_commits[project] = resolve_reset_target(checkout).pinned_commit
        mirrors[project] = ensure_repo_mirror(
            source_checkout=checkout,
            cache_dir=output_dir / CACHE_DIR_NAME,
            repo=project,
            required_commit=pinned_commits[project],
        )

    run_root.mkdir(parents=True, exist_ok=True)
    _write_details(
        run_root / "run_meta.json",
        {
            "label": label,
            "model": model,
            "thinking": thinking,
            "image": runner.image,
            "num": num,
        },
    )

    graded = 0
    errored = 0
    total_cost = 0.0
    for index, entry in enumerate(pending, start=1):
        RichLog.info(
            f"[{index}/{len(pending)}] Grading id={entry.id} "
            f"({entry.project_name}) {entry.qualified_class_name}"
            f"::{entry.method_signature}"
        )
        file_path = derive_java_file_path(
            checkouts[entry.project_name], entry.qualified_class_name
        )
        if file_path is None:
            RichLog.warn(
                f"Could not locate a source file for {entry.qualified_class_name}; "
                "the agent will search the checkout itself."
            )
        result_relpath = f"details/entry-{entry.id:06d}.json"
        details_path = run_root / result_relpath
        spec = build_task_spec(
            entry,
            project_description=project_descriptions[entry.project_name],
            file_path=file_path,
            pinned_commit=pinned_commits[entry.project_name],
            model=model,
            thinking=thinking,
            result_relpath=result_relpath,
        )

        record: dict[str, Any]
        try:
            runner.run_spec(
                spec=spec,
                output_root=run_root,
                spec_path=run_root / "specs" / f"entry-{entry.id:06d}.json",
                repo_source=mirrors[entry.project_name],
            )
            record = json.loads(details_path.read_text(encoding="utf-8"))
        except (DockerSandboxError, OSError, json.JSONDecodeError) as exc:
            record = _synthesized_error_record(entry, model, str(exc))
            _write_details(details_path, record)

        if record.get("status") == "ok" and record.get("grades") is not None:
            try:
                grades = Grades.model_validate(record["grades"])
            except Exception as exc:  # noqa: BLE001 - record-and-continue
                record["status"] = "error"
                record["error"] = f"Invalid grades in container record: {exc}"
                _write_details(details_path, record)
                errored += 1
                RichLog.error(f"id={entry.id}: {record['error']}")
            else:
                aggregate.set_grade(entry.id, grades)
                aggregate.save()
                graded += 1
                RichLog.info(
                    f"id={entry.id}: fidelity={grades.fidelity}, "
                    f"perceived_level={grades.perceived_level}"
                )
        else:
            errored += 1
            RichLog.error(f"id={entry.id}: {record.get('error') or 'Unknown error'}")

        cost = record.get("total_cost_usd")
        if isinstance(cost, (int, float)):
            total_cost += float(cost)

    summary = {
        "graded": graded,
        "errored": errored,
        "skipped": skipped,
        "total_cost_usd": round(total_cost, 4),
        "aggregate_path": str(aggregate.path),
        "details_dir": str(run_root / "details"),
    }
    RichLog.info(
        f"Agent grading complete: graded={graded}, errored={errored}, "
        f"skipped={skipped}, cost=${summary['total_cost_usd']}"
    )
    return summary
