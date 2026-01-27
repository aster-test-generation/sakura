import csv
import json
from pathlib import Path
from typing import Dict, List, Set

# === CONFIGURATION CONSTANTS ===
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SOURCE_DIR = ROOT_DIR / "outputs" / "raw_outputs" / "nl2test_qwen3_output"
MERGE_DIR = ROOT_DIR / "outputs" / "raw_outputs" / "nl2test_qwen3_repair_output"
OUTPUT_DIR = ROOT_DIR / "outputs" / "raw_outputs" / "nl2test_qwen3_output_updated"
TEST2NL_FILE = ROOT_DIR / "resources" / "test2nl" / "filtered_dataset" / "test2nl.csv"
EVAL_FILE_NAME = "nl2test_evaluation_results.json"


def load_test2nl_entries(csv_path: Path) -> Dict[str, Set[int]]:
    """
    Load Test2NL entries from CSV and group IDs by project name.

    Returns:
        Dictionary mapping project_name to set of IDs for that project.
    """
    project_ids: Dict[str, Set[int]] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            project_name = row["project_name"]
            entry_id = int(row["id"])

            if project_name not in project_ids:
                project_ids[project_name] = set()
            project_ids[project_name].add(entry_id)

    return project_ids


def load_evaluation_results(json_path: Path) -> List[dict]:
    """Load NL2TestEval entries from JSON file."""
    if not json_path.exists():
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_ids_from_evaluations(evaluations: List[dict]) -> Set[int]:
    """Extract IDs from evaluation results."""
    ids = set()
    for eval_entry in evaluations:
        nl2test_input = eval_entry.get("nl2test_input", {})
        entry_id = nl2test_input.get("id", -1)
        if entry_id != -1:
            ids.add(entry_id)
    return ids


def save_evaluation_results(json_path: Path, evaluations: List[dict]) -> None:
    """Save evaluation results to JSON file."""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(evaluations, f, indent=2)


def merge_missing_ids() -> None:
    """Main function to merge missing evaluation IDs from MERGE_DIR into OUTPUT_DIR."""
    if not SOURCE_DIR.is_dir():
        print(f"Error: SOURCE_DIR not found: {SOURCE_DIR}")
        return

    if not MERGE_DIR.is_dir():
        print(f"Error: MERGE_DIR not found: {MERGE_DIR}")
        return

    if not TEST2NL_FILE.is_file():
        print(f"Error: TEST2NL_FILE not found: {TEST2NL_FILE}")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Load expected IDs per project from TEST2NL_FILE
    print(f"Loading Test2NL entries from: {TEST2NL_FILE}")
    expected_ids_per_project = load_test2nl_entries(TEST2NL_FILE)
    print(f"Found {len(expected_ids_per_project)} projects in Test2NL file")

    total_added = 0

    # Process each project directory in SOURCE_DIR
    for project_dir in sorted(SOURCE_DIR.iterdir()):
        if not project_dir.is_dir():
            continue

        project_name = project_dir.name
        source_eval_path = project_dir / EVAL_FILE_NAME

        # Skip if project not in Test2NL file
        if project_name not in expected_ids_per_project:
            print(f"Skipping {project_name}: not found in Test2NL file")
            continue

        expected_ids = expected_ids_per_project[project_name]

        # Load source evaluations
        source_evals = load_evaluation_results(source_eval_path)
        source_ids = get_ids_from_evaluations(source_evals)

        # Find missing IDs
        missing_ids = expected_ids - source_ids

        # Create output project directory
        output_project_dir = OUTPUT_DIR / project_name
        output_project_dir.mkdir(parents=True, exist_ok=True)
        output_eval_path = output_project_dir / EVAL_FILE_NAME

        if not missing_ids:
            print(f"{project_name}: No missing IDs")
            # Still copy the source evaluations to OUTPUT_DIR
            save_evaluation_results(output_eval_path, source_evals)
            continue

        print(f"{project_name}: Missing {len(missing_ids)} IDs: {sorted(missing_ids)}")

        # Load merge evaluations
        merge_eval_path = MERGE_DIR / project_name / EVAL_FILE_NAME
        if not merge_eval_path.exists():
            print(f"  Warning: Merge file not found: {merge_eval_path}")
            # Still save source evaluations to OUTPUT_DIR
            save_evaluation_results(output_eval_path, source_evals)
            continue

        merge_evals = load_evaluation_results(merge_eval_path)

        # Find and add missing entries from merge (track added IDs to prevent duplicates)
        entries_added = 0
        added_ids: Set[int] = set()
        for merge_entry in merge_evals:
            nl2test_input = merge_entry.get("nl2test_input", {})
            entry_id = nl2test_input.get("id", -1)

            if entry_id in missing_ids and entry_id not in added_ids:
                source_evals.append(merge_entry)
                added_ids.add(entry_id)
                entries_added += 1
                print(f"  Added ID {entry_id}")

        # Warn about IDs not found in MERGE_DIR
        not_found_ids = missing_ids - added_ids
        if not_found_ids:
            print(f"  Warning: IDs not found in MERGE_DIR: {sorted(not_found_ids)}")

        # Save merged evaluations to OUTPUT_DIR
        save_evaluation_results(output_eval_path, source_evals)
        if entries_added > 0:
            print(f"  Saved {entries_added} new entries to {output_eval_path}")
            total_added += entries_added

    print(f"\nTotal entries added: {total_added}")


def main() -> None:
    merge_missing_ids()


if __name__ == "__main__":
    main()
