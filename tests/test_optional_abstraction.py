from pathlib import Path

import pytest

from sakura.cli import _load_nl2_inputs_by_project_from_csv
from sakura.test2nl.model.models import AbstractionLevel, Test2NLEntry
from sakura.utils.file_io.structured_data_manager import StructuredDataManager


def _csv_entry(abstraction_level: str | None) -> dict[str, object]:
    entry: dict[str, object] = {
        "id": 7,
        "description": "Returns the saved owner.",
        "project_name": "petclinic",
        "qualified_class_name": "example.OwnerTests",
        "method_signature": "testSaveOwner()",
        "is_bdd": False,
    }
    if abstraction_level is not None:
        entry["abstraction_level"] = abstraction_level
    return entry


def test_structured_data_manager_round_trips_optional_abstraction_levels(
    tmp_path: Path,
) -> None:
    manager = StructuredDataManager(tmp_path)
    entries = [
        Test2NLEntry(
            id=7,
            description="Returns the saved owner.",
            project_name="petclinic",
            qualified_class_name="example.OwnerTests",
            method_signature="testSaveOwner()",
            abstraction_level=None,
            is_bdd=False,
        ),
        Test2NLEntry(
            id=8,
            description="Returns the saved owner.",
            project_name="petclinic",
            qualified_class_name="example.OwnerTests",
            method_signature="testSaveOwner()",
            abstraction_level=AbstractionLevel.HIGH,
            is_bdd=False,
        ),
    ]

    manager.save("entries.csv", entries, format="csv")
    loaded = manager.load("entries.csv", Test2NLEntry, format="csv")

    assert loaded[0].abstraction_level is None
    assert loaded[1].abstraction_level is AbstractionLevel.HIGH


@pytest.mark.parametrize("abstraction_level", [None, ""])
def test_csv_conversion_preserves_missing_abstraction_level(
    tmp_path: Path, abstraction_level: str | None
) -> None:
    manager = StructuredDataManager(tmp_path)
    manager.save("entries.csv", _csv_entry(abstraction_level), format="csv")

    grouped = _load_nl2_inputs_by_project_from_csv(
        tmp_path / "entries.csv", max_entries=0, num_proj_parallel=1
    )

    assert grouped["petclinic"][0].abstraction_level is None


def test_csv_conversion_preserves_populated_abstraction_level(tmp_path: Path) -> None:
    manager = StructuredDataManager(tmp_path)
    manager.save("entries.csv", _csv_entry("high"), format="csv")

    grouped = _load_nl2_inputs_by_project_from_csv(
        tmp_path / "entries.csv", max_entries=0, num_proj_parallel=1
    )

    assert grouped["petclinic"][0].abstraction_level == "high"
