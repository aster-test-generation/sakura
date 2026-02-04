"""
Count Test2NL entries that belong to multi-module datasets.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List

from sakura.test2nl.model.models import AbstractionLevel, Test2NLEntry

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TEST2NL_CSV_FILE = "resources/test2nl/filtered_dataset/test2nl.csv"

MULTI_MODULE_DATASETS: dict[str, str] = {
    "commons-email": "multi-module",
    "commons-fileupload": "multi-module",
    "commons-jcs": "multi-module",
    "commons-math": "multi-module",
    "commons-numbers": "multi-module",
    "commons-vfs": "multi-module",
}


def load_test2nl_entries(csv_path: Path) -> List[Test2NLEntry]:
    """Load Test2NL entries from a CSV file."""
    entries: List[Test2NLEntry] = []
    with open(csv_path, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            abstraction_value = row.get("abstraction_level", "").strip()
            abstraction_level = (
                AbstractionLevel(abstraction_value) if abstraction_value else None
            )
            is_bdd_value = row.get("is_bdd", "False").strip().lower()

            entry = Test2NLEntry(
                id=int(row["id"]),
                description=row["description"],
                project_name=row["project_name"],
                qualified_class_name=row["qualified_class_name"],
                method_signature=row["method_signature"],
                abstraction_level=abstraction_level,
                is_bdd=is_bdd_value == "true",
            )
            entries.append(entry)
    return entries


def count_multimodule_entries(
    entries: Iterable[Test2NLEntry],
) -> dict[str, int]:
    """Return a count of entries per multi-module dataset."""
    counts = {name: 0 for name in MULTI_MODULE_DATASETS}
    for entry in entries:
        if entry.project_name in counts:
            counts[entry.project_name] += 1
    return counts


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Count Test2NL entries for multi-module datasets."
    )
    parser.add_argument(
        "--test2nl-file",
        type=Path,
        default=ROOT_DIR / TEST2NL_CSV_FILE,
        help="Path to the Test2NL CSV file.",
    )
    return parser


def main() -> None:
    """Run the multi-module Test2NL entry counter."""
    parser = build_parser()
    args = parser.parse_args()

    csv_path: Path = args.test2nl_file
    if not csv_path.is_file():
        raise FileNotFoundError(f"Test2NL CSV not found: {csv_path}")

    entries = load_test2nl_entries(csv_path)
    counts = count_multimodule_entries(entries)
    total_multimodule = sum(counts.values())

    print(f"Loaded {len(entries)} Test2NL entries")
    print(f"Multi-module entries: {total_multimodule}")
    print("Multi-module entries by project:")
    for project_name in sorted(counts):
        print(f"  {project_name}: {counts[project_name]}")


if __name__ == "__main__":
    main()
