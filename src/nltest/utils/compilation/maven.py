import re
from pathlib import Path
from typing import List, Optional, Tuple, Set

from pydantic import BaseModel, Field

from javabuild.maven_build import MavenBuild

from nltest.utils.analysis import CommonAnalysis

ErrorKey = Tuple[str, int, Optional[int], str]


# TODO: Make a class that is simpler for streamlined LLM output
class CompilationError(BaseModel):
    file: str
    line: int
    column: Optional[int] = None
    message: str
    details: List[str] = Field(default_factory=list, description="Compiler diagnostic details")


class JavaMavenCompilation(MavenBuild):
    """
    Wrapper over javabuild's MavenBuild that adds structured compiler diagnostics.
    """

    DETAIL_SKIP_PREFIXES = (
        "-> [Help",
        "For more information about the errors",
        "Re-run Maven",
    )

    ERROR_HEADER_RE = re.compile(
        r"^\[ERROR\]\s+(?P<path>.*?\.java):\[(?P<line>\d+)(?:,(?P<col>\d+))?\]\s+(?P<msg>.*)$"
    )
    ERROR_INDENT_RE = re.compile(r"^\[ERROR\]\s{2,}\S")

    def __init__(self, project_root: Path):
        super().__init__(str(project_root))

    def get_compilation_errors(self) -> List[CompilationError]:
        """
        Run `mvn test-compile` via MavenBuild and parse every compiler diagnostic.
        """
        compiler_output = self.compile_tests()
        return self.parse_compilation_errors(compiler_output)

    def parse_compilation_errors(self, compiler_output: str) -> List[CompilationError]:
        lines = self._filter_lines(compiler_output)

        errors: List[CompilationError] = []
        seen_keys: Set[ErrorKey] = set()
        current: Optional[CompilationError] = None

        for line in lines:
            header = self.ERROR_HEADER_RE.match(line)
            if header:
                self._flush_current(current, errors, seen_keys)
                raw_path = header.group("path").strip()
                normalized_file = CommonAnalysis.normalize_path_in_project(raw_path, self.project_root)
                column = header.group("col")
                current = CompilationError(
                    file=normalized_file,
                    line=int(header.group("line")),
                    column=int(column) if column is not None else None,
                    message=header.group("msg").strip(),
                )
                continue

            if current and self._is_detail_line(line):
                detail = self._sanitize_detail(line)
                if not self._should_skip_detail(detail):
                    current.details.append(detail)

        self._flush_current(current, errors, seen_keys)
        return errors

    @classmethod
    def _filter_lines(cls, text: str) -> List[str]:
        """Drop noisy lines that never contribute to compiler diagnostics."""
        filtered: List[str] = []
        for raw in text.splitlines():
            line = raw.rstrip("\r\n")
            if line.startswith("[INFO]") and "COMPILATION ERROR" not in line:
                continue
            if line.startswith("Downloading ") or line.startswith("Downloaded "):
                continue
            filtered.append(line)
        return filtered

    @classmethod
    def _is_detail_line(cls, line: str) -> bool:
        """True when the line adds context to the current error block."""
        if cls.ERROR_HEADER_RE.match(line):
            return False
        if line.startswith("[INFO]") or line.startswith("[WARNING]"):
            return False
        if not line.strip():
            return False
        if line.startswith("[ERROR]"):
            return bool(cls.ERROR_INDENT_RE.match(line))
        return line.startswith("  ") or line.startswith("\t")

    @classmethod
    def _sanitize_detail(cls, line: str) -> str:
        """Remove prefixes that Maven adds before the actual diagnostic text."""
        if line.startswith("[ERROR]"):
            line = line[len("[ERROR]"):].lstrip()
        if line.strip() == "^":  # Keep leading spaces if line marks code
            return "^"
        return line.lstrip()

    @classmethod
    def _should_skip_detail(cls, detail: str) -> bool:
        return any(detail.startswith(prefix) for prefix in cls.DETAIL_SKIP_PREFIXES)

    @classmethod
    def _flush_current(
            cls,
            current: Optional[CompilationError],
            errors: List[CompilationError],
            seen_keys: Set[ErrorKey],
    ) -> None:
        if not current:
            return
        key: ErrorKey = (current.file, current.line, current.column, current.message)
        if key not in seen_keys:
            seen_keys.add(key)
            errors.append(current)
