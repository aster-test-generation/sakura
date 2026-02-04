import csv
from pathlib import Path
from typing import List

from sakura.test2nl.model.models import AbstractionLevel, Test2NLEntry

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

FIRST_DATASET_DIR = "resources/test2nl/old_filtered_dataset"
SECOND_DATASET_DIR = "resources/test2nl/missing_dataset"
NEW_DATASET_DIR = "resources/test2nl/filtered_dataset"
TEST2NL_FILE = "test2nl.csv"


def load_test2nl_entries(csv_path: Path) -> List[Test2NLEntry]:
    """Load Test2NL entries from a CSV file."""
    entries: List[Test2NLEntry] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            abstraction_level = None
            if row.get("abstraction_level"):
                abstraction_level = AbstractionLevel(row["abstraction_level"])

            entry = Test2NLEntry(
                id=int(row["id"]),
                description=row["description"],
                project_name=row["project_name"],
                qualified_class_name=row["qualified_class_name"],
                method_signature=row["method_signature"],
                abstraction_level=abstraction_level,
                is_bdd=row.get("is_bdd", "False").lower() == "true",
            )
            entries.append(entry)
    return entries


def save_test2nl_entries(entries: List[Test2NLEntry], csv_path: Path) -> None:
    """Save Test2NL entries to a CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "abstraction_level",
        "description",
        "id",
        "is_bdd",
        "method_signature",
        "project_name",
        "qualified_class_name",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "abstraction_level": (
                        entry.abstraction_level.value if entry.abstraction_level else ""
                    ),
                    "description": entry.description,
                    "id": entry.id,
                    "is_bdd": entry.is_bdd,
                    "method_signature": entry.method_signature,
                    "project_name": entry.project_name,
                    "qualified_class_name": entry.qualified_class_name,
                }
            )


def get_entry_key(entry: Test2NLEntry) -> tuple:
    """Return the unique key for an entry (excluding id and is_bdd)."""
    abstraction = entry.abstraction_level.value if entry.abstraction_level else None
    return (
        entry.project_name,
        entry.qualified_class_name,
        entry.method_signature,
        abstraction,
    )


def main():
    first_csv = ROOT_DIR / FIRST_DATASET_DIR / TEST2NL_FILE
    second_csv = ROOT_DIR / SECOND_DATASET_DIR / TEST2NL_FILE
    output_csv = ROOT_DIR / NEW_DATASET_DIR / TEST2NL_FILE

    if not first_csv.exists():
        raise FileNotFoundError(f"First dataset not found: {first_csv}")
    if not second_csv.exists():
        raise FileNotFoundError(f"Second dataset not found: {second_csv}")

    print(f"Loading first dataset from {first_csv}...")
    first_entries = load_test2nl_entries(first_csv)
    print(f"  Loaded {len(first_entries)} entries")

    print(f"Loading second dataset from {second_csv}...")
    second_entries = load_test2nl_entries(second_csv)
    print(f"  Loaded {len(second_entries)} entries")

    # Build a map from key -> index in first_entries for quick lookup
    first_key_to_idx: dict[tuple, int] = {}
    for idx, entry in enumerate(first_entries):
        first_key_to_idx[get_entry_key(entry)] = idx

    # Separate second entries into replacements and new additions
    replaced_count = 0
    new_entries: List[Test2NLEntry] = []

    for entry in second_entries:
        key = get_entry_key(entry)
        if key in first_key_to_idx:
            # Replace entry in first dataset, keeping the original ID
            idx = first_key_to_idx[key]
            original_id = first_entries[idx].id
            entry.id = original_id
            first_entries[idx] = entry
            replaced_count += 1
        else:
            new_entries.append(entry)

    print(f"  Replaced {replaced_count} entries in first dataset")
    print(f"  Found {len(new_entries)} new unique entries to add")

    # Get max ID from first dataset for appending new entries
    max_id = max(entry.id for entry in first_entries)
    print(f"  Max ID in first dataset: {max_id}")

    # Assign contiguous IDs to new entries
    for i, entry in enumerate(new_entries):
        entry.id = max_id + 1 + i

    # Merge: first dataset (with replacements) + new entries
    merged_entries = first_entries + new_entries
    print(f"  Merged dataset contains {len(merged_entries)} entries")

    # Verify contiguous IDs
    all_ids = sorted(entry.id for entry in merged_entries)
    expected_ids = list(range(all_ids[0], all_ids[-1] + 1))
    if all_ids != expected_ids:
        print("  Warning: IDs are not contiguous")

    print(f"Saving merged dataset to {output_csv}...")
    save_test2nl_entries(merged_entries, output_csv)
    print("  Done!")

    print("\nMerge complete:")
    print(f"  First dataset: {len(first_entries)} entries")
    print(f"  Entries replaced: {replaced_count}")
    print(f"  New entries added: {len(new_entries)}")
    print(f"  Merged dataset: {len(merged_entries)} entries")
    print(f"  Output: {output_csv}")


if __name__ == "__main__":
    main()
