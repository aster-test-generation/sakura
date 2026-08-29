import json
from pathlib import Path
from typing import IO, Any

import pytest

from sakura.utils.file_io.structured_data_manager import StructuredDataManager


def test_json_append_preserves_malformed_file(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    original = b'[{"id": 1}'
    path.write_bytes(original)
    manager = StructuredDataManager(tmp_path)

    with pytest.raises(json.JSONDecodeError):
        manager.save("results.json", {"id": 2}, mode="append")

    assert path.read_bytes() == original


@pytest.mark.parametrize("error_type", [PermissionError, OSError])
def test_json_append_propagates_read_error_without_replacing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[OSError],
) -> None:
    path = tmp_path / "results.json"
    original = b'[{"id": 1}]'
    path.write_bytes(original)
    manager = StructuredDataManager(tmp_path)
    original_open = Path.open

    def fail_target_read(
        self: Path, mode: str = "r", encoding: str | None = None
    ) -> IO[Any]:
        if self == path and mode == "r":
            raise error_type("read denied")
        return original_open(self, mode, encoding=encoding)

    monkeypatch.setattr(Path, "open", fail_target_read)

    with pytest.raises(error_type, match="read denied"):
        manager.save("results.json", {"id": 2}, mode="append")

    assert path.read_bytes() == original


def test_json_save_creates_and_appends_valid_list(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    manager = StructuredDataManager(tmp_path)

    manager.save("results.json", {"id": 1}, mode="append")
    manager.save("results.json", [{"id": 2}, {"id": 3}], mode="append")

    assert json.loads(path.read_text(encoding="utf-8")) == [
        {"id": 1},
        {"id": 2},
        {"id": 3},
    ]


def test_atomic_write_cleans_temporary_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "results.json"
    original = b'[{"id": 1}]'
    path.write_bytes(original)
    replacement_source: Path | None = None

    def fail_replace(self: Path, target: Path) -> Path:
        nonlocal replacement_source
        replacement_source = self
        raise OSError("replace denied")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="replace denied"):
        StructuredDataManager._atomic_write_text(path, '[{"id": 2}]')

    assert path.read_bytes() == original
    assert replacement_source is not None
    assert not replacement_source.exists()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []
