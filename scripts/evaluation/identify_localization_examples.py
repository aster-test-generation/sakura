"""Identify NL2TestEval entries with perfect localization recall (1.0) and save their IDs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError

from sakura.nl2test.models.decomposition import LocalizationEval
from sakura.utils.models.nl2test import NL2TestEval

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = ROOT_DIR / "outputs" / "raw_outputs" / "nl2test_gemini_flash_output"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "decomposition_examples"
EVAL_FILE_NAME = "nl2test_evaluation_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify NL2TestEval entries with perfect localization recall."
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        help="Directory containing project subdirectories with evaluation results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the output JSON file.",
    )
    return parser.parse_args()


def parse_localization_eval(value: Any) -> LocalizationEval | None:
    """Parse a localization eval value into a LocalizationEval object."""
    if value is None:
        return None
    if isinstance(value, LocalizationEval):
        return value
    try:
        return LocalizationEval.model_validate(value)
    except ValidationError:
        return None


def load_eval_file(eval_file: Path) -> List[NL2TestEval]:
    """Load and validate NL2TestEval entries from a JSON file."""
    try:
        with eval_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        print(f"Skipping {eval_file}: invalid JSON.")
        return []

    if not isinstance(data, list):
        print(f"Skipping {eval_file}: expected a list of entries.")
        return []

    results: List[NL2TestEval] = []
    for index, item in enumerate(data):
        try:
            results.append(NL2TestEval.model_validate(item))
        except ValidationError:
            print(f"Skipping entry {index} in {eval_file} due to validation error.")
    return results


def main() -> None:
    args = parse_args()
    eval_dir: Path = args.eval_dir
    output_dir: Path = args.output_dir

    # Normalize paths
    if not eval_dir.is_absolute():
        eval_dir = (ROOT_DIR / eval_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (ROOT_DIR / output_dir).resolve()

    if not eval_dir.exists():
        print(f"Evaluation directory not found: {eval_dir}")
        return

    project_dirs = [path for path in eval_dir.iterdir() if path.is_dir()]
    if not project_dirs:
        print(f"No project directories found in {eval_dir}")
        return

    # Collect IDs with perfect localization recall, grouped by project
    projects_data: Dict[str, Dict[str, Any]] = {}
    total_count = 0

    for project_dir in sorted(project_dirs, key=lambda p: p.name):
        eval_file = project_dir / EVAL_FILE_NAME
        if not eval_file.is_file():
            continue

        project_name = project_dir.name
        entries = load_eval_file(eval_file)
        perfect_ids: List[int] = []

        for entry in entries:
            localization_eval = parse_localization_eval(entry.localization_eval)
            if localization_eval is None:
                continue

            if math.isclose(localization_eval.localization_recall, 1.0):
                perfect_ids.append(entry.nl2test_input.id)

        if perfect_ids:
            projects_data[project_name] = {
                "count": len(perfect_ids),
                "ids": sorted(perfect_ids),
            }
            total_count += len(perfect_ids)

    # Build output JSON
    output_data = {
        "total_count": total_count,
        "projects": projects_data,
    }

    # Write output
    output_file = output_dir / "localization_examples.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(output_data, handle, indent=2)
    except OSError as e:
        print(f"Failed to write output file {output_file}: {e}")
        return

    print(f"Found {total_count} entries with perfect localization recall (1.0)")
    print(f"Results saved to: {output_file}")

    # Print summary by project
    if projects_data:
        print("\nSummary by project:")
        for project_name, data in projects_data.items():
            print(f"  {project_name}: {data['count']} entries")


if __name__ == "__main__":
    main()
