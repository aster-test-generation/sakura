from pathlib import Path
from typing import List

from sakura.utils.compilation.maven import CompilationError, JavaMavenCompilation
from sakura.utils.formatting import ErrorFormatter
from sakura.utils.pretty.color_logger import RichLog
from sakura.utils.vcs.git_utils import GitUtilities

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROJECTS_DIR = ROOT_DIR / "resources" / "datasets"
SUMMARY_SEPARATOR = "=" * 72


def iter_project_dirs(projects_dir: Path) -> List[Path]:
    """Return sorted project directories under the dataset root."""
    if not projects_dir.exists() or not projects_dir.is_dir():
        raise FileNotFoundError(f"Projects directory not found: {projects_dir}")
    return sorted([path for path in projects_dir.iterdir() if path.is_dir()])


def log_compilation_errors(project_name: str, errors: List[CompilationError]) -> None:
    """Log compilation errors for a single project."""
    RichLog.error(f"[{project_name}] {len(errors)} compilation error(s) found.")
    for error in errors:
        RichLog.error(f"[{project_name}] {error.file}")
        RichLog.error(ErrorFormatter.format_compilation_error(error))


def compile_project(project_dir: Path) -> List[CompilationError]:
    """Run Maven test compilation and return structured errors."""
    return JavaMavenCompilation(project_dir).get_compilation_errors()


def log_summary(successes: List[str], failures: List[str]) -> None:
    """Log a formatted summary of compilation results."""
    RichLog.info(SUMMARY_SEPARATOR)
    RichLog.info("Compilation summary")
    RichLog.info(f"Successful projects ({len(successes)}):")
    if successes:
        for name in successes:
            RichLog.info(f"  - {name}")
    else:
        RichLog.info("  (none)")

    RichLog.info(f"Projects with compilation errors ({len(failures)}):")
    if failures:
        for name in failures:
            RichLog.info(f"  - {name}")
    else:
        RichLog.info("  (none)")
    RichLog.info(SUMMARY_SEPARATOR)


def main() -> None:
    RichLog.info(f"Resetting submodules under {PROJECTS_DIR}")
    GitUtilities.reset_submodules_in_dir(PROJECTS_DIR)

    successes: List[str] = []
    failures: List[str] = []

    for project_dir in iter_project_dirs(PROJECTS_DIR):
        project_name = project_dir.name
        RichLog.info(f"Compiling project: {project_name}")
        try:
            errors = compile_project(project_dir)
        except Exception as exc:
            RichLog.error(f"[{project_name}] Compilation failed: {exc}")
            failures.append(project_name)
            continue

        if errors:
            log_compilation_errors(project_name, errors)
            failures.append(project_name)
        else:
            successes.append(project_name)

    log_summary(successes, failures)


if __name__ == "__main__":
    main()
