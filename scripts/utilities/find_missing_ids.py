"""Find IDs from Test2NL dataset that are missing from evaluation results."""

import csv
import json
from pathlib import Path

EVAL_DIR = Path("outputs/raw_outputs/nl2test_qwen3_output_updated/")
TEST2NL_FILE = Path("resources/test2nl/filtered_dataset/test2nl.csv")


def load_test2nl_ids(test2nl_file: Path) -> set[int]:
    """Load all IDs from the Test2NL CSV file."""
    ids = set()
    with open(test2nl_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(int(row["id"]))
    return ids


def load_eval_ids(eval_dir: Path) -> set[int]:
    """Load all IDs from evaluation results across all project directories."""
    ids = set()
    for project_dir in eval_dir.iterdir():
        if not project_dir.is_dir():
            continue
        results_file = project_dir / "nl2test_evaluation_results.json"
        if not results_file.exists():
            continue
        with open(results_file, "r") as f:
            results = json.load(f)
        for result in results:
            nl2test_input = result.get("nl2test_input", {})
            id_val = nl2test_input.get("id", -1)
            if id_val != -1:
                ids.add(id_val)
    return ids


def main():
    test2nl_ids = load_test2nl_ids(TEST2NL_FILE)
    eval_ids = load_eval_ids(EVAL_DIR)

    missing_ids = test2nl_ids - eval_ids
    extra_ids = eval_ids - test2nl_ids

    print(f"Total Test2NL IDs: {len(test2nl_ids)}")
    print(f"Total Eval IDs: {len(eval_ids)}")
    print(f"Missing IDs (in Test2NL but not in Eval): {len(missing_ids)}")
    print(f"Extra IDs (in Eval but not in Test2NL): {len(extra_ids)}")

    if missing_ids:
        print(f"\nMissing IDs: {sorted(missing_ids)}")
    if extra_ids:
        print(f"\nExtra IDs: {sorted(extra_ids)}")


if __name__ == "__main__":
    main()
