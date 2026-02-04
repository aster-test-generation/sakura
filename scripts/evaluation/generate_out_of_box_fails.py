"""Count failed_test_file_generation and failed_code_generation in Gemini CLI results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError

from sakura.utils.models.nl2test import OutOfBoxAgentEval

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_OUTPUTS_DIR = ROOT_DIR / "outputs" / "raw_outputs"
OUTPUT_DIR = ROOT_DIR / "outputs" / "out_of_box_fails"
EVAL_FILE_NAME = "nl2test_evaluation_results.json"


def load_eval_file(eval_file: Path) -> List[OutOfBoxAgentEval]:
    try:
        with eval_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        print(f"Skipping {eval_file}: invalid JSON.")
        return []

    if not isinstance(data, list):
        print(f"Skipping {eval_file}: expected a list of entries.")
        return []

    results: List[OutOfBoxAgentEval] = []
    for index, item in enumerate(data):
        try:
            results.append(OutOfBoxAgentEval.model_validate(item))
        except ValidationError:
            print(f"Skipping entry {index} in {eval_file} due to validation error.")
    return results


def find_gemini_cli_directories() -> List[Path]:
    """Find all directories in raw_outputs starting with 'gemini_cli'."""
    if not RAW_OUTPUTS_DIR.exists():
        print(f"Raw outputs directory not found: {RAW_OUTPUTS_DIR}")
        return []

    return sorted(
        [
            d
            for d in RAW_OUTPUTS_DIR.iterdir()
            if d.is_dir() and d.name.startswith("gemini_cli")
        ]
    )


def process_directory(eval_dir: Path) -> Dict[str, Any]:
    """Process a single Gemini CLI output directory and return failure counts."""
    project_dirs = [p for p in eval_dir.iterdir() if p.is_dir()]

    total_entries = 0
    failed_test_file_generation_count = 0
    failed_code_generation_count = 0
    by_project: Dict[str, Dict[str, int]] = {}

    for project_dir in sorted(project_dirs, key=lambda p: p.name):
        eval_file = project_dir / EVAL_FILE_NAME
        if not eval_file.is_file():
            continue

        entries = load_eval_file(eval_file)
        project_name = project_dir.name

        project_total = 0
        project_failed_test_file = 0
        project_failed_code = 0

        for entry in entries:
            project_total += 1
            if entry.failed_test_file_generation:
                project_failed_test_file += 1
            if entry.failed_code_generation:
                project_failed_code += 1

        total_entries += project_total
        failed_test_file_generation_count += project_failed_test_file
        failed_code_generation_count += project_failed_code

        by_project[project_name] = {
            "total": project_total,
            "failed_test_file_generation": project_failed_test_file,
            "failed_code_generation": project_failed_code,
        }

    return {
        "source_directory": eval_dir.name,
        "total_entries": total_entries,
        "failed_test_file_generation_count": failed_test_file_generation_count,
        "failed_code_generation_count": failed_code_generation_count,
        "by_project": by_project,
    }


def main() -> None:
    gemini_dirs = find_gemini_cli_directories()
    if not gemini_dirs:
        print("No gemini_cli directories found.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for eval_dir in gemini_dirs:
        print(f"Processing: {eval_dir.name}")
        result = process_directory(eval_dir)

        output_file = OUTPUT_DIR / f"{eval_dir.name}.json"
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

        print(f"  Total entries: {result['total_entries']}")
        print(
            f"  Failed test file generation: {result['failed_test_file_generation_count']}"
        )
        print(f"  Failed code generation: {result['failed_code_generation_count']}")
        print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()
