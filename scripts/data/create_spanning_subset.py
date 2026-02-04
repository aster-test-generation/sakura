from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sakura.dataset_creation.model import NL2TestDataset
from sakura.test2nl.model.models import AbstractionLevel, Test2NLEntry
from sakura.utils.file_io.structured_data_manager import StructuredDataManager

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

BUCKETED_DATASET_DIR = ROOT_DIR / "resources/filtered_bucketed_tests"
TEST2NL_FILE = ROOT_DIR / "resources/test2nl/filtered_dataset/test2nl.csv"
OUTPUT_DIR = ROOT_DIR / "resources/test2nl/filtered_dataset"
OUTPUT_FILE_NAME = "spanning_subset.csv"

BUCKET_KEYS = [
    "tests_with_one_focal_methods",
    "tests_with_two_focal_methods",
    "tests_with_more_than_two_to_five_focal_methods",
    "tests_with_more_than_five_to_ten_focal_methods",
    "tests_with_more_than_ten_focal_methods",
]

LEVEL_ORDER = {
    AbstractionLevel.LOW: 0,
    AbstractionLevel.MEDIUM: 1,
    AbstractionLevel.HIGH: 2,
}


@dataclass(frozen=True)
class TestKey:
    project_name: str
    qualified_class_name: str
    method_signature: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.project_name, self.qualified_class_name, self.method_signature)


@dataclass
class Candidate:
    key: TestKey
    bucket: str
    level: AbstractionLevel
    entry: Test2NLEntry

    def coverage_key(self) -> tuple[str, AbstractionLevel]:
        return (self.bucket, self.level)


def iter_project_dataset_paths(bucketed_dir: Path) -> Iterable[tuple[str, Path]]:
    for entry in sorted(bucketed_dir.iterdir()):
        if not entry.is_dir():
            continue
        dataset_path = entry / "nl2test.json"
        if dataset_path.is_file():
            yield entry.name, dataset_path


def normalize_abstraction_level(
    value: AbstractionLevel | str | None,
) -> AbstractionLevel | None:
    if value is None:
        return None
    if isinstance(value, AbstractionLevel):
        return value
    try:
        return AbstractionLevel(str(value).strip().lower())
    except ValueError:
        return None


def load_test2nl_entries(test2nl_file: Path) -> list[Test2NLEntry]:
    manager = StructuredDataManager(test2nl_file.parent)
    return manager.load(test2nl_file.name, Test2NLEntry, format="csv")


def index_test2nl_entries(
    entries: Iterable[Test2NLEntry],
) -> dict[TestKey, dict[AbstractionLevel, Test2NLEntry]]:
    indexed: dict[TestKey, dict[AbstractionLevel, Test2NLEntry]] = defaultdict(dict)
    skipped = 0
    for entry in entries:
        level = normalize_abstraction_level(entry.abstraction_level)
        if level is None:
            skipped += 1
            continue
        key = TestKey(
            entry.project_name, entry.qualified_class_name, entry.method_signature
        )
        indexed[key][level] = entry
    if skipped:
        print(f"[WARN] Skipped {skipped} entries with invalid abstraction_level")
    return indexed


def load_bucketed_tests(bucketed_dir: Path) -> dict[TestKey, str]:
    bucket_by_test: dict[TestKey, str] = {}
    duplicates = 0
    for project_name, dataset_path in iter_project_dataset_paths(bucketed_dir):
        dataset = NL2TestDataset.model_validate_json(
            dataset_path.read_text(encoding="utf-8")
        )
        for bucket in BUCKET_KEYS:
            tests = getattr(dataset, bucket, []) or []
            for test in tests:
                key = TestKey(
                    project_name, test.qualified_class_name, test.method_signature
                )
                if key in bucket_by_test and bucket_by_test[key] != bucket:
                    duplicates += 1
                    continue
                bucket_by_test[key] = bucket
    if duplicates:
        print(f"[WARN] Found {duplicates} tests assigned to multiple buckets")
    return bucket_by_test


def build_candidates(
    bucket_by_test: dict[TestKey, str],
    entries_index: dict[TestKey, dict[AbstractionLevel, Test2NLEntry]],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    missing = 0
    for key, bucket in bucket_by_test.items():
        entries_by_level = entries_index.get(key)
        if not entries_by_level:
            missing += 1
            continue
        for level, entry in entries_by_level.items():
            candidates.append(
                Candidate(key=key, bucket=bucket, level=level, entry=entry)
            )
    if missing:
        print(f"[WARN] {missing} bucketed tests missing from Test2NL CSV")
    return candidates


def required_coverage() -> set[tuple[str, AbstractionLevel]]:
    return {(bucket, level) for bucket in BUCKET_KEYS for level in AbstractionLevel}


def select_spanning_subset(
    candidates: list[Candidate], total_tests: int, max_projects: int
) -> list[Candidate]:
    required = required_coverage()
    available = {c.coverage_key() for c in candidates}

    missing = required - available
    if missing:
        missing_text = ", ".join(
            sorted(f"{bucket}:{level.value}" for bucket, level in missing)
        )
        raise SystemExit(f"[ERROR] Missing coverage in source data: {missing_text}")

    num_combinations = len(required)
    if total_tests < num_combinations:
        raise SystemExit(
            f"[ERROR] total_tests ({total_tests}) must be >= combinations ({num_combinations})"
        )

    # Group candidates by their coverage key (bucket, level)
    candidates_by_coverage: dict[tuple[str, AbstractionLevel], list[Candidate]] = (
        defaultdict(list)
    )
    for candidate in candidates:
        candidates_by_coverage[candidate.coverage_key()].append(candidate)

    # Sort each group for deterministic selection
    for key in candidates_by_coverage:
        candidates_by_coverage[key].sort(key=lambda c: c.key.as_tuple())

    selected: list[Candidate] = []
    used_test_keys: set[TestKey] = set()
    locked_projects: set[str] = set()
    project_counts: dict[str, int] = defaultdict(int)
    coverage_counts: dict[tuple[str, AbstractionLevel], int] = defaultdict(int)

    target_per_project = total_tests // max_projects

    ordered_pairs = sorted(required, key=lambda p: (p[0], LEVEL_ORDER[p[1]]))

    def find_best_candidate(
        coverage_key: tuple[str, AbstractionLevel],
    ) -> Candidate | None:
        candidates_for_pair = candidates_by_coverage[coverage_key]
        best: Candidate | None = None
        best_score: tuple[int, int, str, str, str] | None = None

        projects_at_cap = len(locked_projects) >= max_projects

        for candidate in candidates_for_pair:
            if candidate.key in used_test_keys:
                continue
            project = candidate.key.project_name
            project_count = project_counts[project]
            # If we've hit the project cap, only consider locked projects
            if projects_at_cap and project not in locked_projects:
                continue
            # Skip projects that have already reached their quota
            if project_count >= target_per_project:
                continue
            is_new_project = 1 if project not in locked_projects else 0
            score = (
                -is_new_project,  # Prefer new projects (lower score = better)
                project_count,
                project,
                candidate.key.qualified_class_name,
                candidate.key.method_signature,
            )
            if best_score is None or score < best_score:
                best = candidate
                best_score = score

        return best

    def select_candidate(candidate: Candidate) -> None:
        selected.append(candidate)
        used_test_keys.add(candidate.key)
        locked_projects.add(candidate.key.project_name)
        project_counts[candidate.key.project_name] += 1
        coverage_counts[candidate.coverage_key()] += 1

    # First pass: ensure all 15 combinations are covered with 1 test each
    for coverage_key in ordered_pairs:
        best = find_best_candidate(coverage_key)
        if best is None:
            raise SystemExit(
                f"[ERROR] Cannot find unique test for {coverage_key[0]}:{coverage_key[1].value} "
                f"within {max_projects} projects"
            )
        select_candidate(best)

    # Second pass: fill remaining slots distributed across combinations
    remaining = total_tests - len(selected)
    for _ in range(remaining):
        # Find the combination with the fewest selections that has available candidates
        best_candidate: Candidate | None = None
        best_coverage_count = float("inf")

        for coverage_key in ordered_pairs:
            count = coverage_counts[coverage_key]
            if count >= best_coverage_count:
                continue
            candidate = find_best_candidate(coverage_key)
            if candidate is not None:
                best_candidate = candidate
                best_coverage_count = count

        if best_candidate is None:
            print(
                f"[WARN] Could only select {len(selected)} unique tests (target: {total_tests})"
            )
            break

        select_candidate(best_candidate)

    return selected


def collect_entries(selected: Iterable[Candidate]) -> list[Test2NLEntry]:
    return [candidate.entry for candidate in selected]


def entry_sort_key(entry: Test2NLEntry) -> tuple[str, str, str, int]:
    level = normalize_abstraction_level(entry.abstraction_level)
    level_rank = LEVEL_ORDER[level] if level is not None else 99
    return (
        entry.project_name,
        entry.qualified_class_name,
        entry.method_signature,
        level_rank,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a minimal spanning subset of Test2NL entries covering all buckets "
            "and abstraction levels with a balanced per-project distribution."
        )
    )
    parser.add_argument(
        "--bucketed-dir",
        type=str,
        default=str(BUCKETED_DATASET_DIR),
        help="Path to filtered bucketed tests root.",
    )
    parser.add_argument(
        "--test2nl-file",
        type=str,
        default=str(TEST2NL_FILE),
        help="Path to Test2NL CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for the spanning subset CSV.",
    )
    parser.add_argument(
        "--output-file-name",
        type=str,
        default=OUTPUT_FILE_NAME,
        help="Output CSV file name.",
    )
    parser.add_argument(
        "--total-tests",
        type=int,
        default=20,
        help="Total number of unique tests to select.",
    )
    parser.add_argument(
        "--max-projects",
        type=int,
        default=10,
        help="Maximum number of projects to select tests from (hard cap).",
    )
    args = parser.parse_args()

    bucketed_dir = Path(args.bucketed_dir).expanduser().resolve()
    test2nl_file = Path(args.test2nl_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_file_name = args.output_file_name.strip()
    total_tests = int(args.total_tests)
    max_projects = int(args.max_projects)

    if not bucketed_dir.is_dir():
        raise SystemExit(f"[ERROR] Bucketed dataset dir not found: {bucketed_dir}")
    if not test2nl_file.is_file():
        raise SystemExit(f"[ERROR] Test2NL CSV file not found: {test2nl_file}")
    if not output_file_name:
        raise SystemExit("[ERROR] Output file name must be non-empty")
    if total_tests < 15:
        raise SystemExit(
            "[ERROR] total-tests must be >= 15 (number of bucket/level combinations)"
        )
    if max_projects < 1:
        raise SystemExit("[ERROR] max-projects must be >= 1")

    test2nl_entries = load_test2nl_entries(test2nl_file)
    print(f"[INFO] Loaded {len(test2nl_entries)} Test2NL entries")

    entries_index = index_test2nl_entries(test2nl_entries)
    bucket_by_test = load_bucketed_tests(bucketed_dir)
    candidates = build_candidates(bucket_by_test, entries_index)

    if not candidates:
        raise SystemExit("[ERROR] No candidates found to span buckets/levels")

    selected_candidates = select_spanning_subset(
        candidates, total_tests, max_projects
    )
    selected_entries = collect_entries(selected_candidates)
    selected_entries.sort(key=entry_sort_key)

    project_counts: dict[str, int] = defaultdict(int)
    coverage_counts: dict[tuple[str, AbstractionLevel], int] = defaultdict(int)
    for candidate in selected_candidates:
        project_counts[candidate.key.project_name] += 1
        coverage_counts[candidate.coverage_key()] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    manager = StructuredDataManager(output_dir)
    manager.save(output_file_name, selected_entries, format="csv", mode="write")

    print(
        f"[OK] Selected {len(selected_entries)} unique tests "
        f"from {len(project_counts)} projects"
    )
    print(f"[OK] Wrote to {output_dir / output_file_name}")
    print(f"[INFO] Project distribution: {dict(sorted(project_counts.items()))}")
    coverage_summary = {
        f"{bucket}:{level.value}": coverage_counts[(bucket, level)]
        for bucket in BUCKET_KEYS
        for level in AbstractionLevel
    }
    print(f"[INFO] Coverage distribution: {coverage_summary}")


if __name__ == "__main__":
    main()
