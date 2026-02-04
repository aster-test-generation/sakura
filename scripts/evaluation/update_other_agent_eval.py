"""Re-grade selected evaluation entries based on configurable criteria.

This script loads existing evaluation results from OLD_EVAL_DIR, filters entries
based on criteria (e.g., failed_code_generation), re-grades selected entries,
and writes merged results to NEW_EVAL_DIR.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import ray

# Add scripts directory to path for imports from make_other_agent_eval
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from make_other_agent_eval import (
    build_project_context,
    evaluate_entry,
    index_test2nl_entries,
    load_test2nl_entries,
)

from sakura.test2nl.model.models import AbstractionLevel, Test2NLEntry
from sakura.utils.models import OutOfBoxAgentEval
from sakura.utils.pretty.color_logger import RichLog

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OLD_EVAL_DIR = (
    ROOT_DIR / "outputs" / "raw_outputs" / "gemini_cli_flash_output"
)
NEW_EVAL_DIR = (
    ROOT_DIR / "outputs" / "raw_outputs" / "gemini_cli_flash_output_updated"
)
AGENT_OUTPUT_DIR = ROOT_DIR / "resources" / "outputs" / "gemini_cli_flash_output"
TEST2NL_DIR = ROOT_DIR / "resources" / "test2nl" / "filtered_dataset"
TEST2NL_FILE_NAME = "test2nl.csv"
EVAL_FILE_NAME = "nl2test_evaluation_results.json"

# Re-grading criteria flags
REGRADE_NO_CODE_GENERATION = True  # Re-grade entries with failed_code_generation=True

# Parallelization settings
NUM_PROJ_PARALLEL = 20
TARGET_PROJECTS: list[str] = []  # Empty for all projects


def load_existing_evaluations(
    eval_dir: Path,
) -> dict[str, list[OutOfBoxAgentEval]]:
    """Load existing evaluation results from project subdirectories.

    Args:
        eval_dir: Directory containing project evaluation subdirectories.

    Returns:
        Dict mapping project_name -> list of OutOfBoxAgentEval.
    """
    evaluations: dict[str, list[OutOfBoxAgentEval]] = {}

    if not eval_dir.is_dir():
        RichLog.warn(f"Evaluation directory not found: {eval_dir}")
        return evaluations

    for project_dir in eval_dir.iterdir():
        if not project_dir.is_dir():
            continue

        eval_file = project_dir / EVAL_FILE_NAME
        if not eval_file.is_file():
            RichLog.warn(f"Evaluation file not found: {eval_file}")
            continue

        try:
            with eval_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            evals = [OutOfBoxAgentEval(**entry) for entry in data]
            evaluations[project_dir.name] = evals
            RichLog.info(f"Loaded {len(evals)} evaluations from {project_dir.name}")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            RichLog.error(f"Failed to load {eval_file}: {exc}")

    return evaluations


def filter_entries_for_regrade(evals: list[OutOfBoxAgentEval]) -> set[int]:
    """Filter entries that need re-grading based on configured criteria.

    Args:
        evals: List of existing evaluations.

    Returns:
        Set of entry IDs that need re-grading.
    """
    ids_to_regrade: set[int] = set()

    if REGRADE_NO_CODE_GENERATION:
        for entry in evals:
            if entry.failed_code_generation:
                ids_to_regrade.add(entry.nl2test_input.id)

    return ids_to_regrade


def merge_evaluations(
    original: list[OutOfBoxAgentEval],
    regraded: dict[int, OutOfBoxAgentEval],
) -> list[OutOfBoxAgentEval]:
    """Merge regraded evaluations with original evaluations.

    Args:
        original: Original list of evaluations.
        regraded: Dict mapping entry ID -> regraded OutOfBoxAgentEval.

    Returns:
        Merged list with regraded entries replacing originals.
    """
    merged: list[OutOfBoxAgentEval] = []
    for entry in original:
        entry_id = entry.nl2test_input.id
        if entry_id in regraded:
            merged.append(regraded[entry_id])
        else:
            merged.append(entry)
    return merged


def write_results(
    project_name: str,
    results: list[OutOfBoxAgentEval],
    output_dir: Path,
) -> None:
    """Write evaluation results to output directory.

    Args:
        project_name: Name of the project.
        results: List of evaluation results.
        output_dir: Base output directory.
    """
    project_output_dir = output_dir / project_name
    project_output_dir.mkdir(parents=True, exist_ok=True)
    output_file = project_output_dir / EVAL_FILE_NAME

    with output_file.open("w", encoding="utf-8") as handle:
        json.dump([entry.model_dump() for entry in results], handle, indent=2)

    RichLog.info(f"Saved {len(results)} evaluation results to {output_file}")


@ray.remote
def regrade_project_remote(
    project_name: str,
    entry_lookup: dict[int, dict[str, Any]],
    entry_ids_to_regrade: list[int],
) -> dict[str, Any]:
    """Ray remote function to re-grade selected entries for a project.

    Args:
        project_name: Name of the project.
        entry_lookup: Serializable dict of Test2NL entries.
        entry_ids_to_regrade: List of entry IDs to re-grade.

    Returns:
        Dict with project_name, success flag, and regraded results.
    """
    # Reconstruct Test2NLEntry objects from serializable dict
    entries = {
        entry_id: Test2NLEntry(
            id=data["id"],
            description=data["description"],
            project_name=data["project_name"],
            qualified_class_name=data["qualified_class_name"],
            method_signature=data["method_signature"],
            abstraction_level=(
                AbstractionLevel(data["abstraction_level"])
                if data.get("abstraction_level")
                else None
            ),
            is_bdd=data.get("is_bdd", False),
        )
        for entry_id, data in entry_lookup.items()
    }

    context = build_project_context(project_name)
    if not context:
        return {
            "project_name": project_name,
            "success": False,
            "error": f"Could not build project context for {project_name}",
            "results": {},
        }

    regraded_results: dict[int, dict[str, Any]] = {}
    project_output_dir = AGENT_OUTPUT_DIR / project_name

    for entry_id in entry_ids_to_regrade:
        entry = entries.get(entry_id)
        if not entry:
            RichLog.warn(f"No Test2NL entry found for id={entry_id}")
            continue

        entry_dir = project_output_dir / str(entry_id)
        if not entry_dir.is_dir():
            RichLog.warn(f"Entry directory not found: {entry_dir}")
            continue

        result = evaluate_entry(context, entry, entry_dir)
        regraded_results[entry_id] = result.model_dump()
        RichLog.info(f"Regraded entry {entry_id} for {project_name}")

    return {
        "project_name": project_name,
        "success": True,
        "results": regraded_results,
    }


def main() -> None:
    # Validate that OLD_EVAL_DIR and NEW_EVAL_DIR are different
    if OLD_EVAL_DIR.resolve() == NEW_EVAL_DIR.resolve():
        raise ValueError(
            "OLD_EVAL_DIR and NEW_EVAL_DIR must be different paths. "
            f"Both are set to: {OLD_EVAL_DIR}"
        )

    # Load Test2NL entries
    test2nl_file = TEST2NL_DIR / TEST2NL_FILE_NAME
    if not test2nl_file.is_file():
        raise FileNotFoundError(f"Test2NL CSV not found: {test2nl_file}")

    entries = load_test2nl_entries(test2nl_file)
    entry_lookup = index_test2nl_entries(entries)
    entry_lookup_serializable = {
        entry_id: {
            "id": entry.id,
            "description": entry.description,
            "project_name": entry.project_name,
            "qualified_class_name": entry.qualified_class_name,
            "method_signature": entry.method_signature,
            "abstraction_level": (
                entry.abstraction_level.value if entry.abstraction_level else None
            ),
            "is_bdd": entry.is_bdd,
        }
        for entry_id, entry in entry_lookup.items()
    }

    # Load existing evaluations
    existing_evals = load_existing_evaluations(OLD_EVAL_DIR)
    if not existing_evals:
        RichLog.warn("No existing evaluations found. Exiting.")
        return

    # Filter to target projects if specified
    if TARGET_PROJECTS:
        target_set = set(TARGET_PROJECTS)
        existing_evals = {
            k: v for k, v in existing_evals.items() if k in target_set
        }
        RichLog.info(
            f"TARGET_PROJECTS: processing {len(existing_evals)} projects: {TARGET_PROJECTS}"
        )

    # Identify entries to regrade per project
    projects_to_regrade: dict[str, list[int]] = {}
    projects_to_copy: list[str] = []

    for project_name, evals in existing_evals.items():
        ids_to_regrade = filter_entries_for_regrade(evals)
        if ids_to_regrade:
            projects_to_regrade[project_name] = sorted(ids_to_regrade)
            RichLog.info(
                f"{project_name}: {len(ids_to_regrade)} entries to regrade"
            )
        else:
            projects_to_copy.append(project_name)
            RichLog.info(f"{project_name}: no entries to regrade, will copy as-is")

    # Copy projects with no entries to regrade
    for project_name in projects_to_copy:
        write_results(project_name, existing_evals[project_name], NEW_EVAL_DIR)

    if not projects_to_regrade:
        RichLog.info("No entries to regrade. All results copied to NEW_EVAL_DIR.")
        return

    # Initialize Ray
    try:
        if not ray.is_initialized():
            ray.init()
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Ray: {exc}") from exc

    # Project-level parallel scheduling
    pending_projects = deque(projects_to_regrade.keys())
    inflight_futures: set[ray.ObjectRef] = set()
    future_to_project: dict[ray.ObjectRef, str] = {}

    total_regraded = 0

    def launch_projects_up_to_limit() -> None:
        while len(inflight_futures) < NUM_PROJ_PARALLEL and pending_projects:
            project_name = pending_projects.popleft()
            entry_ids = projects_to_regrade[project_name]
            RichLog.info(
                f"Starting regrade for project: {project_name} ({len(entry_ids)} entries)"
            )
            fut = regrade_project_remote.remote(
                project_name, entry_lookup_serializable, entry_ids
            )
            inflight_futures.add(fut)
            future_to_project[fut] = project_name

    launch_projects_up_to_limit()

    # Main scheduling loop
    while inflight_futures or pending_projects:
        if not inflight_futures:
            launch_projects_up_to_limit()
            if not inflight_futures and not pending_projects:
                break

        ready_refs, _ = ray.wait(list(inflight_futures), num_returns=1)
        for ref in ready_refs:
            project_name = future_to_project.pop(ref, None)
            inflight_futures.discard(ref)
            if project_name is None:
                continue

            try:
                result: dict[str, Any] = ray.get(ref)
            except Exception as exc:
                RichLog.error(f"[{project_name}] Regrade failed: {exc}")
                # Copy original results on failure
                write_results(project_name, existing_evals[project_name], NEW_EVAL_DIR)
                launch_projects_up_to_limit()
                continue

            if not result.get("success"):
                RichLog.warn(f"[{project_name}] {result.get('error', 'Unknown error')}")
                # Copy original results on failure
                write_results(project_name, existing_evals[project_name], NEW_EVAL_DIR)
                launch_projects_up_to_limit()
                continue

            regraded_data = result.get("results", {})
            if regraded_data:
                regraded_evals = {
                    entry_id: OutOfBoxAgentEval(**data)
                    for entry_id, data in regraded_data.items()
                }
                merged = merge_evaluations(
                    existing_evals[project_name], regraded_evals
                )
                write_results(project_name, merged, NEW_EVAL_DIR)
                total_regraded += len(regraded_data)
                RichLog.info(
                    f"[{project_name}] Completed regrade: {len(regraded_data)} entries updated"
                )
            else:
                # No entries regraded, copy original
                write_results(project_name, existing_evals[project_name], NEW_EVAL_DIR)
                RichLog.warn(f"[{project_name}] No entries were regraded")

            launch_projects_up_to_limit()

    # Shutdown Ray
    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

    RichLog.info(
        f"Regrade complete: {total_regraded} entries regraded across "
        f"{len(projects_to_regrade)} projects"
    )


if __name__ == "__main__":
    main()
