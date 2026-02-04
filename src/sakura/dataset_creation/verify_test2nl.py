import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

from sakura.dataset_creation.model import NL2TestDataset, Test

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

TEST2NL_DIR = "resources/test2nl/filtered_dataset"
TEST2NL_FILE = "test2nl.csv"
BUCKETED_TESTS_DIR = "resources/filtered_bucketed_tests"
BUCKETED_FILE = "nl2test.json"

REQUIRED_ABSTRACTION_LEVELS = {"low", "medium", "high"}


def load_test2nl_ids_and_entries(
    csv_path: Path,
) -> Tuple[List[int], Dict[Tuple[str, str, str], Set[str]]]:
    """
    Load Test2NL entries from CSV and return:
    1. List of all IDs in order
    2. Dict mapping (project_name, qualified_class_name, method_signature)
       to set of abstraction levels
    """
    ids: List[int] = []
    entries: Dict[Tuple[str, str, str], Set[str]] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row["id"]))
            key = (
                row["project_name"],
                row["qualified_class_name"],
                row["method_signature"],
            )
            abstraction_level = row.get("abstraction_level", "")
            if key not in entries:
                entries[key] = set()
            if abstraction_level:
                entries[key].add(abstraction_level)

    return ids, entries


def verify_contiguous_ids(ids: List[int]) -> Tuple[bool, List[str]]:
    """Verify IDs start at 0 and are contiguous with no gaps."""
    errors: List[str] = []

    if not ids:
        errors.append("No IDs found in dataset")
        return False, errors

    sorted_ids = sorted(ids)

    if sorted_ids[0] != 0:
        errors.append(f"IDs do not start at 0 (first ID: {sorted_ids[0]})")

    expected_ids = list(range(sorted_ids[0], sorted_ids[-1] + 1))
    if sorted_ids != expected_ids:
        missing = set(expected_ids) - set(sorted_ids)
        duplicates = [id_ for id_ in sorted_ids if sorted_ids.count(id_) > 1]
        if missing:
            errors.append(
                f"Missing IDs: {sorted(missing)[:20]}{'...' if len(missing) > 20 else ''}"
            )
        if duplicates:
            unique_dups = sorted(set(duplicates))
            errors.append(
                f"Duplicate IDs: {unique_dups[:20]}{'...' if len(unique_dups) > 20 else ''}"
            )

    return len(errors) == 0, errors


def get_all_tests(dataset: NL2TestDataset) -> List[Test]:
    """Return all tests from all buckets."""
    return (
        dataset.tests_with_one_focal_methods
        + dataset.tests_with_two_focal_methods
        + dataset.tests_with_more_than_two_to_five_focal_methods
        + dataset.tests_with_more_than_five_to_ten_focal_methods
        + dataset.tests_with_more_than_ten_focal_methods
    )


def load_bucketed_methods(bucketed_dir: Path) -> Set[Tuple[str, str, str]]:
    """Load all method keys from all bucketed test datasets."""
    methods: Set[Tuple[str, str, str]] = set()

    for project_name in sorted(os.listdir(bucketed_dir)):
        if project_name.startswith(".") or project_name.startswith("__"):
            continue

        project_path = bucketed_dir / project_name
        if not project_path.is_dir():
            continue

        bucketed_file = project_path / BUCKETED_FILE
        if not bucketed_file.exists():
            continue

        with open(bucketed_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = NL2TestDataset(**data)

        for test in get_all_tests(dataset):
            key = (project_name, test.qualified_class_name, test.method_signature)
            methods.add(key)

    return methods


def verify_abstraction_levels(
    test2nl_entries: Dict[Tuple[str, str, str], Set[str]],
    bucketed_methods: Set[Tuple[str, str, str]],
) -> Tuple[bool, List[str], Dict[str, int]]:
    """Verify all bucketed methods have all three abstraction levels in Test2NL."""
    errors: List[str] = []
    stats = {
        "total_methods": len(bucketed_methods),
        "complete": 0,
        "missing_completely": 0,
        "incomplete_levels": 0,
    }

    missing_methods: List[Tuple[str, str, str]] = []
    incomplete_methods: List[Tuple[Tuple[str, str, str], Set[str]]] = []

    for method_key in bucketed_methods:
        if method_key not in test2nl_entries:
            stats["missing_completely"] += 1
            missing_methods.append(method_key)
        elif test2nl_entries[method_key] != REQUIRED_ABSTRACTION_LEVELS:
            stats["incomplete_levels"] += 1
            incomplete_methods.append((method_key, test2nl_entries[method_key]))
        else:
            stats["complete"] += 1

    if missing_methods:
        errors.append(
            f"Methods completely missing from Test2NL: {stats['missing_completely']}"
        )
        for method in missing_methods[:5]:
            errors.append(f"  - {method[0]}:{method[1]}#{method[2]}")
        if len(missing_methods) > 5:
            errors.append(f"  ... and {len(missing_methods) - 5} more")

    if incomplete_methods:
        errors.append(
            f"Methods with incomplete abstraction levels: {stats['incomplete_levels']}"
        )
        for method, levels in incomplete_methods[:5]:
            missing_levels = REQUIRED_ABSTRACTION_LEVELS - levels
            errors.append(
                f"  - {method[0]}:{method[1]}#{method[2]} (missing: {missing_levels})"
            )
        if len(incomplete_methods) > 5:
            errors.append(f"  ... and {len(incomplete_methods) - 5} more")

    is_valid = stats["missing_completely"] == 0 and stats["incomplete_levels"] == 0
    return is_valid, errors, stats


def main():
    test2nl_file = ROOT_DIR / TEST2NL_DIR / TEST2NL_FILE
    bucketed_dir = ROOT_DIR / BUCKETED_TESTS_DIR

    if not test2nl_file.exists():
        raise FileNotFoundError(f"Test2NL file not found: {test2nl_file}")
    if not bucketed_dir.exists():
        raise FileNotFoundError(f"Bucketed tests directory not found: {bucketed_dir}")

    print(f"Loading Test2NL from {test2nl_file}...")
    ids, test2nl_entries = load_test2nl_ids_and_entries(test2nl_file)
    print(f"  Loaded {len(ids)} entries ({len(test2nl_entries)} unique methods)")

    print(f"\nLoading bucketed tests from {bucketed_dir}...")
    bucketed_methods = load_bucketed_methods(bucketed_dir)
    print(f"  Loaded {len(bucketed_methods)} methods")

    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)

    # Verify contiguous IDs
    print("\n1. Checking ID contiguity...")
    ids_valid, id_errors = verify_contiguous_ids(ids)
    if ids_valid:
        print(f"   PASS: IDs are contiguous (0 to {max(ids)})")
    else:
        print("   FAIL:")
        for error in id_errors:
            print(f"   - {error}")

    # Verify abstraction levels
    print("\n2. Checking abstraction level completeness...")
    levels_valid, level_errors, stats = verify_abstraction_levels(
        test2nl_entries, bucketed_methods
    )
    if levels_valid:
        print(
            f"   PASS: All {stats['total_methods']} methods have all abstraction levels"
        )
    else:
        print("   FAIL:")
        for error in level_errors:
            print(f"   {error}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total Test2NL entries: {len(ids)}")
    print(f"  Unique methods in Test2NL: {len(test2nl_entries)}")
    print(f"  Methods in bucketed dataset: {stats['total_methods']}")
    print(f"  Complete (all 3 levels): {stats['complete']}")
    print(f"  Missing completely: {stats['missing_completely']}")
    print(f"  Incomplete levels: {stats['incomplete_levels']}")

    all_valid = ids_valid and levels_valid
    print(f"\nOverall: {'PASS' if all_valid else 'FAIL'}")

    return 0 if all_valid else 1


if __name__ == "__main__":
    exit(main())
