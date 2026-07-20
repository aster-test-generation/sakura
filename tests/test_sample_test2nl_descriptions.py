import csv
from pathlib import Path

import pytest

from sakura.dataset_creation.sample_test2nl_descriptions import (
    create_abstraction_samples,
)


FIELDNAMES = [
    "abstraction_level",
    "description",
    "id",
    "is_bdd",
    "method_signature",
    "project_name",
    "qualified_class_name",
]


def _write_input_csv(path: Path, entries_per_level: int = 3) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        row_id = 0
        for level in ("low", "medium", "high"):
            for index in range(entries_per_level):
                writer.writerow(
                    {
                        "abstraction_level": level,
                        "description": f"{level} description {index}",
                        "id": row_id,
                        "is_bdd": False,
                        "method_signature": f"test{index}()",
                        "project_name": "example-project",
                        "qualified_class_name": "example.ExampleTest",
                    }
                )
                row_id += 1


def test_writes_one_csv_per_level(tmp_path: Path) -> None:
    input_csv = tmp_path / "test2nl.csv"
    output_dir = tmp_path / "sample"
    _write_input_csv(input_csv)

    written_files = create_abstraction_samples(
        2, output_dir, input_csv=input_csv, seed=7
    )

    assert {path.name for path in written_files.values()} == {
        "low.csv",
        "medium.csv",
        "high.csv",
    }
    for level in ("low", "medium", "high"):
        with (output_dir / f"{level}.csv").open(
            "r", encoding="utf-8", newline=""
        ) as csv_file:
            rows = list(csv.DictReader(csv_file))
        assert len(rows) == 2
        assert {row["abstraction_level"] for row in rows} == {level}


def test_reproducible_with_seed(tmp_path: Path) -> None:
    input_csv = tmp_path / "test2nl.csv"
    _write_input_csv(input_csv)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    create_abstraction_samples(2, first_output, input_csv=input_csv, seed=11)
    create_abstraction_samples(2, second_output, input_csv=input_csv, seed=11)

    for level in ("low", "medium", "high"):
        first_csv = first_output / f"{level}.csv"
        second_csv = second_output / f"{level}.csv"
        assert first_csv.read_text() == second_csv.read_text()


def test_rejects_oversized_sample(tmp_path: Path) -> None:
    input_csv = tmp_path / "test2nl.csv"
    _write_input_csv(input_csv, entries_per_level=1)

    with pytest.raises(ValueError, match="only 1 are available"):
        create_abstraction_samples(2, tmp_path / "sample", input_csv=input_csv)
