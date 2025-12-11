from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, NamedTuple

import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel

# Root directory (project root, two levels up from this script)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Directory paths relative to ROOT_DIR
PROJECTS_DIR = "resources/datasets/"
BASE_ANALYSIS_DIR = "resources/analysis/"
OVERRIDE_EXISTING = True


class AnalysisResult(NamedTuple):
    project: str
    status: str
    message: str | None = None


def iter_project_dirs(projects_root: Path) -> Iterable[Path]:
    """Yield immediate child directories beneath the projects root."""
    for entry in sorted(projects_root.iterdir()):
        if entry.is_dir():
            yield entry


@ray.remote
def generate_analysis_for_project(
        project_path: str, analysis_dir: str, override_existing: bool
) -> AnalysisResult:
    """
    Generate or refresh a CLDK analysis for a single project.
    """
    project_root = Path(project_path)
    project_name = project_root.name
    target_dir = Path(analysis_dir) / project_name
    target_dir.mkdir(parents=True, exist_ok=True)

    analysis_json = target_dir / "analysis.json"

    if analysis_json.exists() and not override_existing:
        return AnalysisResult(project_name, "skipped", "analysis.json already present")

    eager = override_existing or not analysis_json.exists()

    try:
        cldk = CLDK(language="java")
        cldk.analysis(
            project_path=str(project_root),
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=target_dir,
            eager=eager,
        )
        outcome = "overwritten" if analysis_json.exists() and override_existing else "generated"
        return AnalysisResult(project_name, outcome)
    except Exception as exc:
        return AnalysisResult(project_name, "failed", str(exc))


def main() -> None:
    projects_root = (ROOT_DIR / PROJECTS_DIR).resolve()
    analysis_root = (ROOT_DIR / BASE_ANALYSIS_DIR).resolve()

    if not projects_root.is_dir():
        raise FileNotFoundError(f"Projects directory not found: {projects_root}")

    analysis_root.mkdir(parents=True, exist_ok=True)

    projects = list(iter_project_dirs(projects_root))

    if not projects:
        print(f"No projects found under {projects_root}. Nothing to do.", flush=True)
        return

    print(f"Initializing Ray to process {len(projects)} projects...", flush=True)
    ray.init(ignore_reinit_error=True)

    try:
        remote_tasks = [
            generate_analysis_for_project.remote(
                str(project_path),
                str(analysis_root),
                OVERRIDE_EXISTING,
            )
            for project_path in projects
        ]

        results = ray.get(remote_tasks)
    finally:
        ray.shutdown()

    has_failures = False

    for result in results:
        line = f"[{result.status.upper():9}] {result.project}"
        if result.message:
            line += f" - {result.message}"
        print(line, flush=True)
        if result.status == "failed":
            has_failures = True

    if has_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
