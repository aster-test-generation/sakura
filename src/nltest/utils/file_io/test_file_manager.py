from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Annotated, Optional, Union

from pydantic import BaseModel
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nltest.nl2test.models import NL2TestInput
    from nltest.test2nl.model.models import RoundTripTest, AbstractionLevel


class TestFileInfo(BaseModel):
    """
    Container for single generated test file.
    """

    qualified_class_name: Annotated[
        str, "The qualified class name of the developer-written test"
    ]
    method_signature: Annotated[
        str, "The method signature of the developer-written test"
    ]
    description: Annotated[str, "The description of the developer-written test"] = ""
    test_code: Annotated[str, "The test code of the autonomously generated test"] = ""
    abstraction_level: Annotated[
        Optional[AbstractionLevel],
        "The abstraction level (from Test2NL) of the natural language description of the developer-written test",
    ] = None
    trial_number: Annotated[
        Optional[int], "The current trial number (for evaluation)"
    ] = 1
    id: Annotated[Optional[int], "The ID from NL2TestInput"] = -1

    @classmethod
    def from_nl2test_input(
        cls,
        nl2test_input: NL2TestInput,
        *,
        test_code: str = "",
        trial_number: int = -1,
    ) -> "TestFileInfo":
        return cls(
            qualified_class_name=nl2test_input.qualified_class_name,
            method_signature=nl2test_input.method_signature,
            description=nl2test_input.description,
            test_code=test_code,
            abstraction_level=nl2test_input.abstraction_level,
            trial_number=trial_number,
            id=nl2test_input.id,
        )

    @classmethod
    def from_roundtrip_test(
        cls,
        rt_test: RoundTripTest,
    ) -> "TestFileInfo":
        return cls(
            qualified_class_name=rt_test.qualified_class_name,
            method_signature=rt_test.method_signature,
            description=rt_test.generated_description.description,
            test_code=rt_test.generated_test,
            abstraction_level=rt_test.generated_description.abstraction_level,
            trial_number=rt_test.generated_description.trial_number,
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
    def decode_file_name(file_name: str) -> Tuple[str, Optional[AbstractionLevel], int]:
        # Remove .java extension if present
        if file_name.endswith(".java"):
            file_name = file_name[:-5]

        class_name = file_name.split("/")[-1]
        return TestFileManager.decode_class_name(class_name)

    def target_path(self, test_info: TestFileInfo, *, encode_class_name: bool = True) -> Path:
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

        # Non-encoded: Use the fully qualified class name as path components
        qcn_path = Path(*test_info.qualified_class_name.split("."))
        return self.test_base_dir / (qcn_path.with_suffix(".java"))

    def _rewrite_java_header(self, test_info: TestFileInfo, code: str) -> str:
        """
        Ensures the Java package declaration and the class name are correctly assigned.
        """
        package = test_info.qualified_class_name.rsplit(".", 1)[0]
        encoded_name = self.encode_class_name(
            test_info.id,
        )

        if package:
            pkg_decl = f"package {package};"
            pkg_regex = r"(?m)^\s*package\s+[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*;"
            if re.search(pkg_regex, code):
                code = re.sub(pkg_regex, pkg_decl, code, count=1)
            else:
                code = pkg_decl + "\n\n" + code.lstrip()
        else:
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
        code = re.sub(top_level_decl, rf"\g<prefix>{encoded_name}", code, count=1)

        return code

    def save_single(
        self,
        test_info: TestFileInfo,
        *,
        sync_names: bool = False,
        encode_class_name: bool = False,
    ) -> Path:
        """
        Save a single test file. Optionally encode the class name for the filename.

        - If `encode_class_name` is True, the file will be saved using the
          encoded class name at the package directory derived from the
          qualified class name.
        - If `encode_class_name` is False, the file will be saved mirroring the
          qualified class name as a path rooted at `self.test_base_dir`.

        TODO: Handle conflicts more gracefully (e.g., versioning or merging).
        """
        file_path = self.target_path(test_info, encode_class_name=encode_class_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Conflict detection: avoid overwriting existing files for now
        if file_path.exists():
            # TODO: Decide on conflict resolution strategy beyond raising an exception.
            raise Exception(
                f"Conflicting test file already exists at {file_path}."
            )

        content = test_info.test_code
        if sync_names and encode_class_name:
            # Only rewrite headers when using encoded class names
            content = self._rewrite_java_header(test_info, content)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path

    def save_batch(self, tests: List[TestFileInfo]) -> List[Path]:
        saved = []
        for t in tests:
            # Preserve legacy behavior for batch saves: use encoded class names
            saved.append(self.save_single(t, encode_class_name=True))
        return saved

    def load(self, test_info: TestFileInfo) -> str:
        file_path = self.target_path(test_info)
        if not file_path.exists():
            raise FileNotFoundError(f"Test file not found at {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def make_test_fqn(self, test_info: TestFileInfo) -> str:
        class_name = self.encode_class_name(test_info.id)
        package = test_info.qualified_class_name.rsplit(".", 1)[0]
        return f"{package}.{class_name}" if package else class_name

    def delete_single(
        self, test_info: TestFileInfo, *, encode_class_name: bool = False
    ) -> Path:
        """
        Delete the test file at the location where it would have been saved.

        - If `encode_class_name` is True, uses the encoded class name path.
        - If `encode_class_name` is False, deletes the file at the fully
          qualified class name path.
        """
        file_path = self.target_path(test_info, encode_class_name=encode_class_name)
        if file_path.exists():
            file_path.unlink()
        return file_path
