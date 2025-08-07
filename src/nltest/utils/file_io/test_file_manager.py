import re
from pathlib import Path
from typing import List, Tuple, Annotated, Optional

from pydantic import BaseModel

from nltest.test2nl.model.models import RoundTripTest, AbstractionLevel


class TestFileInfo(BaseModel):
    qualified_class_name: Annotated[str, "The qualified class name of the developer-written test"]
    method_signature: Annotated[str, "The method signature of the developer-written test"]
    description: Annotated[str, "The description of the developer-written test"]
    test_code: Annotated[str, "The test code of the autonomously generated test"] = ""
    abstraction_level: Annotated[Optional[
        AbstractionLevel], "The abstraction level (from Test2NL) of the natural language description of the developer-written test"] = None
    trial_number: Annotated[Optional[int], "The current trial number (for evaluation)"] = 1


class TestFileManager:
    def __init__(self, project_root: Path):
        project_root.mkdir(parents=True, exist_ok=True)
        self.project_root = project_root
        self.test_base_dir = project_root / "src" / "test" / "java"

    @staticmethod
    def convert_roundtrip_to_file_info(roundtrip_tests: List[RoundTripTest]) -> List[TestFileInfo]:
        test_file_infos: List[TestFileInfo] = []
        for rt_test in roundtrip_tests:
            test_file_infos.append(TestFileInfo(
                qualified_class_name=rt_test.qualified_class_name,
                method_signature=rt_test.method_signature,
                description=rt_test.generated_description.description,
                test_code=rt_test.generated_test,
                abstraction_level=rt_test.generated_description.abstraction_level,
                trial_number=rt_test.generated_description.trial_number,
            ))
        return test_file_infos

    @staticmethod
    def encode_class_name(method_signature: str, abstraction_level: AbstractionLevel | None = None,
                          trial_number: int = 1) -> str:
        abstraction_level = abstraction_level.value if abstraction_level else "None"

        method_sig_pattern = re.compile(r"\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
        match = method_sig_pattern.search(method_signature)
        if not match:
            raise ValueError(f"Could not parse method signature: {method_signature}")
        method_name = match.group(1)

        return f"NL2Test_{method_name}_{abstraction_level}_{trial_number}"

    def save_single(self, test_info: TestFileInfo) -> None:
        qualified_class_name = test_info.qualified_class_name
        package_path = qualified_class_name.rsplit(".", 1)[0]
        qualified_class_path = Path(*package_path.split("."))
        output_dir = self.test_base_dir / qualified_class_path
        output_dir.mkdir(parents=True, exist_ok=True)

        code = test_info.test_code

        # Replace class name with encoded class name
        class_name = self.encode_class_name(test_info.method_signature, test_info.abstraction_level,
                                            test_info.trial_number)
        class_name_pattern = re.compile(r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\{")
        class_match = class_name_pattern.search(code)
        if not class_match:
            raise ValueError("Could not find class definition in test code.")
        old_class_name = class_match.group(1)
        code = code.replace(f"class {old_class_name}", f"class {class_name}", 1)

        code = f"package {package_path};\n\n{code}"

        file_path = output_dir / f"{class_name}.java"
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(code)

    def save(self, test_infos: List[TestFileInfo]) -> None:
        for test_info in test_infos:
            self.save_single(test_info)

    def load(self, test_info: TestFileInfo) -> str:
        qualified_class_name = test_info.qualified_class_name
        package_path = qualified_class_name.rsplit(".", 1)[0]
        qualified_class_path = Path(*package_path.split("."))
        output_dir = self.test_base_dir / qualified_class_path

        class_name = self.encode_class_name(test_info.method_signature, test_info.abstraction_level,
                                            test_info.trial_number)
        file_path = output_dir / f"{class_name}.java"

        if not file_path.exists():
            raise FileNotFoundError(f"Test file not found at {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
