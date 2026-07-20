"""Create random, per-abstraction samples from the filtered Test2NL dataset."""

from __future__ import annotations

import csv
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT / "resources" / "test2nl" / "filtered_dataset" / "test2nl.csv"
)
ABSTRACTION_LEVELS = ("low", "medium", "high")


def create_abstraction_samples(
    num_each_abstraction: int,
    output_dir: str | Path,
    *,
    input_csv: str | Path = DEFAULT_INPUT_CSV,
    seed: int | None = None,
) -> dict[str, Path]:
    """Sample Test2NL rows and save one CSV for each abstraction level.

    Args:
        num_each_abstraction: Number of rows to sample for every level.
        output_dir: Directory in which ``low.csv``, ``medium.csv``, and
            ``high.csv`` will be written.
        input_csv: Source Test2NL CSV. Exposed for tests; the CLI uses the
            repository's filtered dataset.
        seed: Optional random seed for reproducible snapshots.

    Returns:
        A mapping from abstraction level to its output CSV path.
    """
    if num_each_abstraction <= 0:
        raise ValueError("num_each_abstraction must be greater than zero")

    input_path = Path(input_csv)
    if not input_path.is_file():
        raise FileNotFoundError(f"Test2NL CSV not found: {input_path}")

    with input_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"Test2NL CSV has no header: {input_path}")
        fieldnames = reader.fieldnames
        entries = list(reader)

    if "abstraction_level" not in fieldnames:
        message = "Test2NL CSV is missing the abstraction_level column: "
        raise ValueError(f"{message}{input_path}")

    entries_by_level = {level: [] for level in ABSTRACTION_LEVELS}
    for entry in entries:
        level = entry["abstraction_level"]
        if level in entries_by_level:
            entries_by_level[level].append(entry)

    for level, level_entries in entries_by_level.items():
        if len(level_entries) < num_each_abstraction:
            raise ValueError(
                f"Requested {num_each_abstraction} {level} entries, "
                f"but only {len(level_entries)} are available"
            )

    rng = random.Random(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written_files: dict[str, Path] = {}

    for level in ABSTRACTION_LEVELS:
        available_entries = entries_by_level[level]
        sampled_entries = rng.sample(available_entries, num_each_abstraction)
        file_path = output_path / f"{level}.csv"
        with file_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sampled_entries)
        written_files[level] = file_path

    return written_files
