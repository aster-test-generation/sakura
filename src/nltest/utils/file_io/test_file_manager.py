from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, List, Optional, Tuple, Union

from pydantic import BaseModel

from nltest.utils.exceptions.tool_exceptions import FileDeletionError
from nltest.utils.pretty.color_logger import RichLog

if TYPE_CHECKING:
    from nltest.nl2test.models import NL2TestInput
    from nltest.test2nl.model.models import RoundTripTest


class TestFileInfo(BaseModel):
    """
    Container for single generated test file.
    """

    qualified_class_name: Annotated[str, "The qualified class name of the test"]
    test_code: Annotated[str, "The test code of the test"] = ""
    id: Annotated[Optional[int], "The ID from NL2TestInput"] = -1

    @classmethod
    def from_nl2test_input(
        cls,
        nl2test_input: NL2TestInput,
        *,
        test_code: str = "",
    ) -> "TestFileInfo":
        return cls(
            qualified_class_name=nl2test_input.qualified_class_name,
            test_code=test_code,
            id=nl2test_input.id,
        )

    @classmethod
    def from_roundtrip_test(
        cls,
        rt_test: RoundTripTest,
    ) -> "TestFileInfo":
        return cls(
            qualified_class_name=rt_test.qualified_class_name,
            test_code=rt_test.generated_test,
            id=-1,  # Default value for roundtrip test
        )


class TestFileManager:
    def __init__(
        self, project_root: Path, test_base_dir: Union[str, Path] = "src/test/java"
    ):
        project_root.mkdir(parents=True, exist_ok=True)
        self.project_root = project_root
        self.test_base_dir = project_root / Path(test_base_dir)

    @staticmethod
    def _atomic_write(target_path: Path, content: str) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Optional[str] = None
        fd: Optional[int] = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(target_path.parent),
                prefix=f".{target_path.name}.",
                suffix=".tmp",
            )
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                tmp_file.write(content)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                fd = None  # fd handled by context manager
            os.replace(tmp_path, target_path)
            dir_fd: Optional[int] = None
            if hasattr(os, "O_DIRECTORY"):
                try:
                    dir_fd = os.open(str(target_path.parent), os.O_DIRECTORY)
                except OSError:
                    dir_fd = None
            if dir_fd is not None:
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    @staticmethod
    def _sanitize_generated_code(code: str) -> str:
        if not isinstance(code, str):
            return code

        s = code.strip("\ufeff\r\n\t ")

        # Remove a leading ``` (optionally with language) or leading ''' / """
        s = re.sub(r"^\s*(?:```[^\n]*\n|(?:'''|\"\"\")[\t ]*\n?)", "", s)

        # Remove a trailing ```, ''' or """
        s = re.sub(r"(?:\n?```|\n?(?:'''|\"\"\"))\s*$", "", s)

        return s

    @staticmethod
    def _package_dir_from_qualified(qualified_class_name: str) -> Path:
        package = qualified_class_name.rsplit(".", 1)[0]
        return Path(*package.split(".")) if package else Path()

    @staticmethod
    def encode_class_name(id: int = -1) -> str:
        """
        Produce an encoded Java class name for a generated test case.
        E.g., NL2T_001 for ID-based encoding
        """
        if id == -1:
            raise ValueError("ID-based encoding requires a valid ID (id != -1)")

        class_name = f"NL2T_{id:03d}"
        return class_name

    @staticmethod
    def decode_class_name(encoded_class_name: str) -> int:
        """
        Decode the class name to an ID.
        """
        # Remove the "NL2T_" prefix if present
        if encoded_class_name.startswith("NL2T_"):
            encoded_class_name = encoded_class_name[5:]

        # Only support ID-based encoding
        if not encoded_class_name.isdigit():
            raise ValueError(
                f"Invalid encoded class name format: expected numeric ID, got '{encoded_class_name}'. Legacy method signature encoding is no longer supported."
            )

        return int(encoded_class_name)

    @staticmethod
    def decode_file_name(file_name: str) -> int:
        # Remove .java extension if present
        if file_name.endswith(".java"):
            file_name = file_name[:-5]

        class_name = file_name.split("/")[-1]
        return TestFileManager.decode_class_name(class_name)

    def target_path(
        self, test_info: TestFileInfo, *, encode_class_name: bool = True
    ) -> Path:
        """
        Compute the target file path for a test file.
        - If encode_class_name is True, use the encoded class name (e.g., NL2T_001.java)
          under the package path derived from qualified_class_name.
        - If encode_class_name is False, place the file directly under
          `self.test_base_dir` mirroring the fully qualified class name
          as a path, with `.java` appended at the end.
        """
        if encode_class_name:
            class_name = self.encode_class_name(test_info.id)
            return (
                self.test_base_dir
                / self._package_dir_from_qualified(test_info.qualified_class_name)
                / f"{class_name}.java"
            )
        else:
            # Non-encoded: Use the fully qualified class name as path components
            qcn_path = Path(*test_info.qualified_class_name.split("."))
            return self.test_base_dir / (qcn_path.with_suffix(".java"))

    def _rewrite_java_header(self, package: str, new_class_name: str, code: str) -> str:
        """
        Ensure the Java package declaration and top-level type name match the target.

        - Adds or replaces the package declaration with `package` (if non-empty),
          or removes it if package is empty.
        - Rewrites the first top-level class/interface/enum/record name to `new_class_name`.
        """
        if package:
            pkg_decl = f"package {package};"
            pkg_regex = r"(?m)^\s*package\s+[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*;"
            if re.search(pkg_regex, code):
                code = re.sub(pkg_regex, pkg_decl, code, count=1)
            else:
                code = pkg_decl + "\n\n" + code.lstrip()
        else:
            # Remove any existing package declaration
            code = re.sub(
                r"(?m)^\s*package\s+[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*;\s*\n?",
                "",
                code,
                count=1,
            )

        top_level_decl = (
            r"(?m)^(?P<prefix>\s*"
            r"(?:@\w+(?:\([^)]*\))?\s*)*"
            r"(?:public\s+)?"
            r"(?:abstract\s+|final\s+)?"
            r"(?:class|interface|enum|record)\s+)"
            r"(?P<name>[A-Za-z_]\w*)"
        )
        code = re.sub(top_level_decl, rf"\g<prefix>{new_class_name}", code, count=1)

        return code

    def save_single(
        self,
        test_info: TestFileInfo,
        *,
        sync_names: bool = False,
        encode_class_name: bool = False,
        sanitize_wrappers: bool = True,
        allow_overwrite: bool = False,
    ) -> Tuple[str, Path]:
        """
        Save a single test file. If a conflict occurs, append a numeric suffix
        to the class name (starting at 1) until a free filename is found unless
        `allow_overwrite` is True, in which case the existing file is atomically
        replaced.
        """
        # Determine package and base class name
        if encode_class_name:
            base_class_name = self.encode_class_name(test_info.id)
            package = test_info.qualified_class_name.rsplit(".", 1)[0]
        else:
            if "." in test_info.qualified_class_name:
                package, base_class_name = test_info.qualified_class_name.rsplit(".", 1)
            else:
                package, base_class_name = "", test_info.qualified_class_name

        # Compute parent directory and initial file path
        parent_dir = self.test_base_dir / (
            self._package_dir_from_qualified(package + ".Dummy" if package else "")
        )
        parent_dir.mkdir(parents=True, exist_ok=True)

        # Add number to the end until a nonconflict unless overwriting in place
        class_name = base_class_name
        file_path = parent_dir / f"{class_name}.java"
        if not allow_overwrite:
            counter = 1
            while file_path.exists():
                class_name = f"{base_class_name}{counter}"
                file_path = parent_dir / f"{class_name}.java"
                counter += 1

        # Optionally sanitize wrapper noise before any processing
        content = (
            self._sanitize_generated_code(test_info.test_code)
            if sanitize_wrappers
            else test_info.test_code
        )

        # Rewrite package and class name in code content
        conflict_renamed = class_name != base_class_name
        if sync_names or conflict_renamed:
            content = self._rewrite_java_header(package, class_name, content)

        # Write file
        self._atomic_write(file_path, content)

        # Return the final qualified class name and path
        qualified_name = f"{package}.{class_name}" if package else class_name
        return qualified_name, file_path

    def save_batch(self, tests: List[TestFileInfo]) -> List[Path]:
        saved_paths: List[Path] = []
        for t in tests:
            # Preserve legacy behavior for batch saves: use encoded class names
            _, p = self.save_single(t, encode_class_name=True)
            saved_paths.append(p)
        return saved_paths

    def load(self, test_info: TestFileInfo, *, encode_class_name: bool = False) -> str:
        file_path = self.target_path(test_info, encode_class_name=encode_class_name)
        if not file_path.exists():
            raise FileNotFoundError(f"Test file not found at {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def make_test_fqn(self, test_info: TestFileInfo) -> str:
        class_name = self.encode_class_name(test_info.id)
        package = test_info.qualified_class_name.rsplit(".", 1)[0]
        return f"{package}.{class_name}" if package else class_name

    def delete_single(
        self,
        test_info: TestFileInfo,
        *,
        encode_class_name: bool = False,
        strict: bool = False,
        max_attempts: int = 2,
        retry_delay: float = 0.05,
    ) -> bool:
        """
        Delete the test file at the location where it would have been saved.

        - If `encode_class_name` is True, uses the encoded class name path.
        - If `encode_class_name` is False, deletes the file at the fully
          qualified class name path.

        Returns True if the file existed and was successfully deleted, False otherwise.
        When `strict` is True, an exception is raised if the file cannot be removed
        after the configured retry attempts.
        """
        file_path = self.target_path(test_info, encode_class_name=encode_class_name)
        if not file_path.exists():
            RichLog.info(f"File does not exist, skipping delete: {file_path}")
            return True if strict else False

        attempts = 0
        last_error: Optional[Exception] = None
        while attempts < max_attempts:
            attempts += 1
            try:
                file_path.unlink()
            except FileNotFoundError:
                break
            except Exception as exc:
                last_error = exc
                RichLog.error(f"Failed to delete {file_path}: {exc}")
                if attempts >= max_attempts:
                    break
                time.sleep(max(retry_delay, 0.0))
            else:
                break

        if file_path.exists():
            if strict:
                raise FileDeletionError(
                    f"Failed to delete test file at {file_path}",
                    extra_info={
                        "path": str(file_path),
                        "attempts": attempts,
                        "error": str(last_error) if last_error else "",
                    },
                )
            return False

        return True
