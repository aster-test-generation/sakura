"""Update Gemini CLI evaluation files with tool_log from metadata files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError

from sakura.utils.models.nl2test import OutOfBoxAgentEval

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESOURCES_DIR = ROOT_DIR / "resources"
RAW_OUTPUTS_DIR = ROOT_DIR / "outputs" / "raw_outputs"

EVAL_FILE_NAME = "nl2test_evaluation_results.json"

GEMINI_CLI_CONFIGS: List[Dict[str, Path]] = [
    {
        "eval_dir": RAW_OUTPUTS_DIR / "gemini_cli_pro_output",
        "metadata_dir": RESOURCES_DIR / "outputs" / "gemini_cli_pro_output",
        "output_dir": RAW_OUTPUTS_DIR / "gemini_cli_pro_output_updated",
    },
    {
        "eval_dir": RAW_OUTPUTS_DIR / "gemini_cli_flash_output",
        "metadata_dir": RESOURCES_DIR / "outputs" / "gemini_cli_flash_output",
        "output_dir": RAW_OUTPUTS_DIR / "gemini_cli_flash_output_updated",
    },
]


def load_metadata_tool_counts(
    metadata_dir: Path, project_name: str, entry_id: int
) -> Dict[str, int] | None:
    """Load tool_calls_by_name from metadata.json for a specific entry."""
    metadata_file = (
        metadata_dir / project_name / str(entry_id) / "run_001" / "metadata.json"
    )
    if not metadata_file.is_file():
        return None
    try:
        with metadata_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        tokens = data.get("tokens", {})
        return tokens.get("tool_calls_by_name", {})
    except (json.JSONDecodeError, KeyError):
        return None


def build_tool_log(tool_counts: Dict[str, int]) -> Dict[str, Any]:
    """Build a ToolLog structure with tool counts in supervisor_tool_log."""
    empty_agent_log = {"tool_counts": {}, "tool_trajectories": []}
    return {
        "supervisor_tool_log": {
            "tool_counts": tool_counts,
            "tool_trajectories": [],
        },
        "localization_tool_log": empty_agent_log,
        "composition_tool_log": empty_agent_log,
    }


def process_config(config: Dict[str, Path]) -> None:
    """Process one Gemini CLI configuration and update entries with tool_log."""
    eval_dir = config["eval_dir"]
    metadata_dir = config["metadata_dir"]
    output_dir = config["output_dir"]

    if not eval_dir.exists():
        print(f"Eval directory not found: {eval_dir}")
        return
    if not metadata_dir.exists():
        print(f"Metadata directory not found: {metadata_dir}")
        return

    print(f"\nProcessing: {eval_dir.name}")
    print(f"  Metadata source: {metadata_dir}")
    print(f"  Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_entries = 0
    updated_entries = 0
    missing_metadata = 0

    for project_dir in sorted(eval_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        eval_file = project_dir / EVAL_FILE_NAME
        if not eval_file.is_file():
            continue

        project_name = project_dir.name
        try:
            with eval_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"  Skipping {project_name}: invalid JSON")
            continue

        if not isinstance(data, list):
            print(f"  Skipping {project_name}: expected list")
            continue

        updated_data: List[Dict[str, Any]] = []
        for item in data:
            total_entries += 1
            try:
                entry = OutOfBoxAgentEval.model_validate(item)
            except ValidationError:
                updated_data.append(item)
                continue

            entry_id = entry.nl2test_input.id
            tool_counts = load_metadata_tool_counts(
                metadata_dir, project_name, entry_id
            )

            if tool_counts is not None:
                item["tool_log"] = build_tool_log(tool_counts)
                updated_entries += 1
            else:
                missing_metadata += 1

            updated_data.append(item)

        project_output_dir = output_dir / project_name
        project_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = project_output_dir / EVAL_FILE_NAME
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(updated_data, f, indent=2)

    print(f"  Total entries: {total_entries}")
    print(f"  Updated with tool_log: {updated_entries}")
    print(f"  Missing metadata: {missing_metadata}")


def main() -> None:
    for config in GEMINI_CLI_CONFIGS:
        process_config(config)
    print("\nDone.")


if __name__ == "__main__":
    main()
