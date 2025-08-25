import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

# Get directories relative to this script
PROJECTS_ROOT: Path = Path("../tests/resources")
OUTPUT_BASE: Path = Path("../tests/output")
# Other script configuration options
EVALUATE: bool = False
CLEAR_DATASET: bool = True
MAX_WORKERS: Optional[int] = 1

def process_project(project_dir: Path, src_dir: Path, output_base: Path, evaluate: bool, clear_dataset: bool) -> None:
    project_name = project_dir.name
    output_dir = (output_base / project_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "nltest.cli",
        "generate-descriptions",
        "--project-root",
        str(project_dir.resolve()),
        "--output-dir",
        str(output_dir),
    ]
    if evaluate:
        cmd.append("--evaluate")
    if clear_dataset:
        cmd.append("--clear-dataset")

    print(f"Processing '{project_name}'...", flush=True)
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=src_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed on {project_name}", flush=True)
        print(f"Return code: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
    else:
        print(f"Done with {project_name}", flush=True)
        if result.stdout:
            print(result.stdout)


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    src_dir = (script_dir / ".." / "src").resolve()
    projects_root = (script_dir / PROJECTS_ROOT).resolve()
    output_base = (script_dir / OUTPUT_BASE).resolve()

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir} (expected ../src)")

    cli_file = src_dir / "nltest" / "cli.py"
    if not cli_file.is_file():
        raise FileNotFoundError(f"CLI not found at expected path: {cli_file}")

    if not projects_root.is_dir():
        raise FileNotFoundError(f"Projects root not found: {projects_root}")

    all_projects = sorted([p for p in projects_root.iterdir() if p.is_dir()])
    print(f"Creating Test2NL for {len(all_projects)} total project(s)...", flush=True)

    default_workers = min(len(all_projects), os.cpu_count() or 1) or 1
    max_workers = MAX_WORKERS if (MAX_WORKERS and MAX_WORKERS > 0) else default_workers

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_project, proj_dir, src_dir, output_base, EVALUATE, CLEAR_DATASET)
            for proj_dir in all_projects
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error during parallel execution: {e}", flush=True)

    print("Completed description generation...", flush=True)


if __name__ == "__main__":
    main()
