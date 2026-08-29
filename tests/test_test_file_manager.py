from __future__ import annotations

from pathlib import Path

import pytest

from sakura.utils.exceptions.tool_exceptions import FileDeletionError
from sakura.utils.file_io.test_file_manager import TestFileInfo, TestFileManager


@pytest.fixture
def test_file_info() -> TestFileInfo:
    return TestFileInfo(
        qualified_class_name="example.GeneratedTest",
        test_code="public class GeneratedTest {}",
    )


@pytest.mark.parametrize("strict", [False, True])
def test_delete_single_returns_false_for_missing_file(
    tmp_path: Path,
    test_file_info: TestFileInfo,
    *,
    strict: bool,
) -> None:
    manager = TestFileManager(tmp_path)

    deleted = manager.delete_single(test_file_info, strict=strict)

    assert deleted is False


def test_delete_single_removes_existing_file(
    tmp_path: Path, test_file_info: TestFileInfo
) -> None:
    manager = TestFileManager(tmp_path)
    file_path = manager.target_path(test_file_info)
    file_path.parent.mkdir(parents=True)
    file_path.write_text(test_file_info.test_code, encoding="utf-8")

    deleted = manager.delete_single(test_file_info, strict=True)

    assert deleted is True
    assert not file_path.exists()


def test_delete_single_returns_false_after_persistent_unlink_failure(
    tmp_path: Path,
    test_file_info: TestFileInfo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TestFileManager(tmp_path)
    file_path = manager.target_path(test_file_info)
    file_path.parent.mkdir(parents=True)
    file_path.write_text(test_file_info.test_code, encoding="utf-8")
    unlink_attempts = 0

    def fail_unlink(path: Path, missing_ok: bool = False) -> None:
        nonlocal unlink_attempts
        if path == file_path:
            unlink_attempts += 1
            raise PermissionError("deletion denied")
        original_unlink(path, missing_ok=missing_ok)

    original_unlink = Path.unlink
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    deleted = manager.delete_single(
        test_file_info,
        strict=False,
        max_attempts=3,
        retry_delay=0,
    )

    assert deleted is False
    assert unlink_attempts == 3
    assert file_path.exists()


def test_delete_single_raises_after_persistent_unlink_failure_in_strict_mode(
    tmp_path: Path,
    test_file_info: TestFileInfo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TestFileManager(tmp_path)
    file_path = manager.target_path(test_file_info)
    file_path.parent.mkdir(parents=True)
    file_path.write_text(test_file_info.test_code, encoding="utf-8")
    unlink_attempts = 0

    def fail_unlink(path: Path, missing_ok: bool = False) -> None:
        nonlocal unlink_attempts
        if path == file_path:
            unlink_attempts += 1
            raise PermissionError("deletion denied")
        original_unlink(path, missing_ok=missing_ok)

    original_unlink = Path.unlink
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(FileDeletionError) as exc_info:
        manager.delete_single(
            test_file_info,
            strict=True,
            max_attempts=3,
            retry_delay=0,
        )

    assert unlink_attempts == 3
    assert file_path.exists()
    assert exc_info.value.extra_info == {
        "path": str(file_path),
        "attempts": 3,
        "error": "deletion denied",
    }
