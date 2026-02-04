import os
import sys
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

from pydantic import BaseModel, Field

from sakura.utils.constants import MAVEN_CMD


class CompilationError(BaseModel):

    file: str
    line: int
    column: Optional[int] = None
    message: str
    details: List[str] = Field(default_factory=list)


class JavaCompilation:
    """
    Compile and prepare Maven-based Java project for testing, and parse compiler errors.
    """

    @staticmethod
    def _normalize_project_root(project_root: Union[str, Path]) -> Tuple[Path, Path]:
        """
        Returns (abs_project_root, pom_path) and verifies pom.xml exists.
        """
        abs_root = Path(project_root).expanduser().resolve()
        pom = abs_root / "pom.xml"
        if not pom.is_file():
            raise FileNotFoundError(f"pom.xml not found at {pom}")
        return abs_root, pom

    @staticmethod
    def _resolve_maven_cmd_parts(project_root: Union[str, Path]) -> List[str]:
        """
        Resolves the correct Maven command: mvnw wrapper if present (with sh on *nix if needed),
        otherwise MAVEN_CMD on PATH. Returns parts ready to prepend to args.
        """
        wrapper = "mvnw.cmd" if sys.platform == "win32" else "mvnw"
        root = Path(project_root)
        wrapper_path = root / wrapper
        if wrapper_path.is_file():
            if sys.platform != "win32" and not os.access(wrapper_path, os.X_OK):
                return ["sh", str(wrapper_path)]
            return [str(wrapper_path)]
        mvn_path = shutil.which(MAVEN_CMD)
        if mvn_path:
            return [mvn_path]
        raise FileNotFoundError(
            f"Maven not found. Neither '{MAVEN_CMD}' on PATH nor wrapper '{wrapper}' at {project_root}."
        )

    @staticmethod
    def _run_mvn(
        project_root: Union[str, Path], cmd_base: List[str], args: List[str]
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd_base + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(
                Path(project_root)
            ),  # Ensure relative paths resolve from the project root
        )

    @staticmethod
    def _run_compile(
        project_root: Union[str, Path], pom_path: Union[str, Path], cmd_base: List[str]
    ) -> subprocess.CompletedProcess:
        # Use absolute -f path and batch mode; avoid color for easier parsing
        return JavaCompilation._run_mvn(
            project_root,
            cmd_base,
            [
                "-f",
                str(Path(pom_path).resolve()),
                "-B",
                "-Dstyle.color=never",
                "clean",
                "test-compile",
            ],
        )

    @staticmethod
    def _should_try_formatting(stdout: str, stderr: str) -> bool:
        text = f"{stdout or ''}\n{stderr or ''}".lower()
        triggers = (
            "spring-javaformat",
            "spring-javaformat:check",
            "spring javaformat",
            "format violation",
            "please run 'mvn spring-javaformat:apply'",
        )
        return any(t in text for t in triggers)

    @staticmethod
    def _apply_format(
        project_root: Union[str, Path], pom_path: Union[str, Path], cmd_base: List[str]
    ) -> bool:
        res = JavaCompilation._run_mvn(
            project_root,
            cmd_base,
            ["-f", str(Path(pom_path).resolve()), "spring-javaformat:apply"],
        )
        return res.returncode == 0

    @staticmethod
    def _compile_with_auto_format(
        project_root: Union[str, Path],
    ) -> Tuple[subprocess.CompletedProcess, str]:
        """
        Performs compilation (clean test-compile). If formatting is requested by plugins,
        applies it and re-compiles. Returns (CompletedProcess, combined_output).
        """
        abs_root, pom = JavaCompilation._normalize_project_root(project_root)
        cmd_base = JavaCompilation._resolve_maven_cmd_parts(abs_root)

        result = JavaCompilation._run_compile(abs_root, pom, cmd_base)

        if result.returncode != 0 and JavaCompilation._should_try_formatting(
            result.stdout, result.stderr
        ):
            if JavaCompilation._apply_format(abs_root, pom, cmd_base):
                result = JavaCompilation._run_compile(abs_root, pom, cmd_base)

        combined = f"{result.stdout or ''}\n{result.stderr or ''}"
        # Some callers provide quoted blobs in tests; strip single quote fence if present.
        if len(combined) >= 2 and combined[0] == "'" and combined[-1] == "'":
            combined = combined[1:-1]
        return result, combined

    @staticmethod
    def _report_path(filepath: str, project_root: Optional[Union[str, Path]]) -> str:
        """
        Returns a modified path with no user prefix.
        """
        if not filepath:
            return filepath
        norm_file = filepath.replace("\\", "/")
        if project_root:
            pr = str(Path(project_root).expanduser().resolve()).replace("\\", "/")
            pr_with_sep = pr + "/"
            # Case-insensitive startswith for Windows paths
            if norm_file.lower().startswith(pr_with_sep.lower()):
                return norm_file[len(pr_with_sep) :]
        # Try to trim to portion under /src/
        idx = norm_file.find("/src/")
        if idx != -1:
            return norm_file[idx + 1 :]
        return os.path.basename(norm_file)

    @staticmethod
    def _iter_lines(text_or_lines: Union[str, List[str]]) -> List[str]:
        if isinstance(text_or_lines, str):
            lines = text_or_lines.splitlines()
        else:
            lines = list(text_or_lines)
        # Normalize newlines and strip right side
        return [ln.rstrip("\r\n") for ln in lines]

    @staticmethod
    def _is_new_error_line(line: str) -> bool:
        """
        True if the line introduces a new compiler error for a .java source file.
        """
        return bool(re.match(r"^\[ERROR\]\s+.*?\.java:\[\d+(?:,\d+)?\]\s+.*", line))

    @staticmethod
    def _is_detail_line(line: str) -> bool:
        """
        Lines that are part of the current error explanation. Maven sometimes prefixes them with [ERROR],
        sometimes they are bare indented lines.
        """
        # Examples:
        # "  symbol:   class MockitoBean"
        # "  location: package org.springframework.boot.test.mock.mockito"
        # "[ERROR]   symbol: class DisabledInNativeImage"
        # "[ERROR]     method org.springframework.... is not applicable"
        # "    method org.springframework.... is not applicable"
        if JavaCompilation._is_new_error_line(line):
            return False
        if line.startswith("[INFO]") or line.startswith("[WARNING]"):
            return False
        stripped = line.lstrip()
        if not stripped:
            return False
        if line.startswith("[ERROR]"):
            # Keep indented [ERROR] detail lines (two or more spaces after tag).
            return bool(re.match(r"^\[ERROR\]\s{2,}\S", line))
        # Bare indented line
        return line.startswith("  ") or line.startswith("\t")

    @staticmethod
    def _parse_error_header(line: str) -> Optional[Dict[str, Union[str, int]]]:
        """
        Parses an error header line like:
        [ERROR] /path/to/Foo.java:[12,34] cannot find symbol
        or
        [ERROR] C:\p\Foo.java:[8,2] error: package x.y does not exist
        Returns dict with file, line, column, message (raw).
        """
        m = re.match(
            r"^\[ERROR\]\s+(?P<path>.*?\.java):\[(?P<line>\d+)(?:,(?P<col>\d+))?\]\s+(?P<msg>.*)$",
            line,
        )
        if not m:
            return None
        path = m.group("path").strip()
        line_no = int(m.group("line"))
        col = m.group("col")
        col_no = int(col) if col is not None else None
        msg = m.group("msg").strip()
        return {"path": path, "line": line_no, "column": col_no, "message": msg}

    @staticmethod
    def _sanitize_detail(line: str) -> str:
        """
        Removes [ERROR] prefix and leading spaces, keeps core content.
        """
        if line.startswith("[ERROR]"):
            line = line[len("[ERROR]") :].lstrip()
        return line.lstrip()

    @staticmethod
    def _parse_errors_from_lines(
        lines: List[str], project_root: Optional[Union[str, Path]]
    ) -> Tuple[List[str], List[CompilationError]]:
        """
        Core parser. Walks the lines once, builds a set of erroneous files (.java), and
        a structured list of compiler errors with important details.
        """
        error_classes: set = set()
        errors: List[CompilationError] = []
        current: Optional[CompilationError] = None
        seen_keys: set = set()

        for raw in lines:
            line = raw.rstrip()

            # Start of a new error entry
            if JavaCompilation._is_new_error_line(line):
                header = JavaCompilation._parse_error_header(line)
                if header:
                    if current is not None:
                        key = (
                            current.file,
                            current.line,
                            current.column,
                            current.message,
                        )
                        if key not in seen_keys:
                            seen_keys.add(key)
                            errors.append(current)
                    file_for_report = JavaCompilation._report_path(
                        header["path"], project_root
                    )
                    filename = os.path.basename(header["path"].replace("\\", "/"))
                    if filename.endswith(".java"):
                        error_classes.add(filename)
                    current = CompilationError(
                        file=file_for_report,
                        line=header["line"],
                        column=header["column"],
                        message=header["message"],
                    )
                continue

            # If we are inside an error block, collect detail lines that matter
            if current is not None and JavaCompilation._is_detail_line(line):
                detail = JavaCompilation._sanitize_detail(line)
                if detail.startswith("-> [Help"):
                    continue
                if detail.startswith("For more information about the errors"):
                    continue
                if detail.startswith("Re-run Maven"):
                    continue
                current.details.append(detail)
                continue

            # Pass lines without closing error until we get a new header

        if current is not None:
            key = (
                current.file,
                current.line,
                current.column,
                current.message,
            )
            if key not in seen_keys:
                seen_keys.add(key)
                errors.append(current)

        return sorted(error_classes), errors

    @staticmethod
    def parse_compilation_errors(
        text_or_lines: Union[str, List[str]],
        project_root: Optional[Union[str, Path]] = None,
    ) -> Tuple[List[str], List[CompilationError]]:
        """
        Parse Maven compiler output (string or list of lines), remove noise like INFO lines,
        and return (erroneous_files, errors) without running Maven. The first value is a sorted
        list of .java filenames with compilation errors, not FQCNs.

        Example:
            (['MyTest.java'], [CompilationError(file='src/test/java/com/acme/MyTest.java', line=12, column=8, message='cannot find symbol', details=['symbol:   class Foo', 'location: class com.acme.MyTest'])])
        """
        lines = JavaCompilation._iter_lines(text_or_lines)
        cleaned: List[str] = []
        for ln in lines:
            if ln.startswith("[INFO]"):
                if "COMPILATION ERROR" not in ln:
                    continue
            if ln.startswith("Downloading ") or ln.startswith("Downloaded "):
                continue
            cleaned.append(ln)
        return JavaCompilation._parse_errors_from_lines(cleaned, project_root)

    @staticmethod
    def get_erroneous_files_and_errors(
        project_root: Union[str, Path],
    ) -> Tuple[List[str], List[CompilationError]]:
        """
        Run Maven (clean test-compile), auto-apply Spring JavaFormat if requested,
        and return both the erroneous files and the parsed compilation errors.

        Returns a tuple (erroneous_files, errors) where erroneous_files is a sorted list of
        Java source filenames with compilation errors (e.g., ['MyTest.java']).
        """
        _, combined = JavaCompilation._compile_with_auto_format(project_root)
        return JavaCompilation.parse_compilation_errors(combined, project_root)

    @staticmethod
    def get_erroneous_files(project_root: Union[str, Path]) -> List[str]:
        files, _errors = JavaCompilation.get_erroneous_files_and_errors(project_root)
        return files


if __name__ == "__main__":
    # Example usage (will run Maven in the given project root):
    project_root = "tests/resources/spring-petclinic/"
    error_files, errors = JavaCompilation.get_erroneous_files_and_errors(
        project_root
    )
    print("Erroneous files:", error_files)
    print("Errors:")
    for i, e in enumerate(errors, 1):
        col = f",{e.column}" if e.column is not None else ""
        print(f"{i}. {e.file}:{e.line}{col} -> {e.message}")
        for d in e.details:
            print(f"   - {d}")
