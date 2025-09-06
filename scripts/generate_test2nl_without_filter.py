import sys
import subprocess
from pathlib import Path

# === CONFIGURATION CONSTANTS ===
# Directory paths relative to this script
SRC_DIR = "../src"
BASE_PROJECT_DIR = "../tests/resources"
OUTPUT_DIR = "../tests/output"

# CLI arguments
CLEAR_DATASET = True
MAX_ENTRIES = 0  # Note: 0 = unlimited
LLM_MODEL = "google/gemini-2.0-flash-001"
ONLY_INTERESTING_TESTS = True


def run_generate_descriptions_for_entire_projects(
    base_project_dir: Path,
    src_dir: Path,
    output_dir: Path,
    clear_dataset: bool,
    max_entries: int,
    llm_model: str | None = None,
    only_interesting_tests: bool = False,
) -> None:
    cmd = [
        "poetry",
        "run",
        "python",
        "-u",
        "-m",
        "nltest.cli",
        "generate-descriptions-for-entire-projects",
        "--base-project-dir",
        str(base_project_dir),
        "--output-dir",
        str(output_dir),
    ]
    if clear_dataset:
        cmd.append("--clear-dataset")
    if max_entries > 0:
        cmd.extend(["--max-entries", str(max_entries)])
    if llm_model:
        cmd.extend(["--llm-model", llm_model])
    if only_interesting_tests:
        cmd.append("--only-interesting-tests")

    print(
        f"Running description generation (entire projects) in {base_project_dir}...",
        flush=True,
    )
    print(f"Command: {' '.join(cmd)}", flush=True)

    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=src_dir,
        )
        print("Description generation completed successfully!", flush=True)
    except subprocess.CalledProcessError as e:
        print("Description generation failed!", flush=True)
        print(f"Return code: {e.returncode}")
        sys.exit(1)


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    # Set up paths using configuration constants
    src_dir = (script_dir / SRC_DIR).resolve()
    base_project_dir = (script_dir / BASE_PROJECT_DIR).resolve()
    output_dir = (script_dir / OUTPUT_DIR).resolve()

    # Verify paths exist
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    cli_file = src_dir / "nltest" / "cli.py"
    if not cli_file.is_file():
        raise FileNotFoundError(f"CLI not found at expected path: {cli_file}")

    if not base_project_dir.is_dir():
        raise FileNotFoundError(f"Base project directory not found: {base_project_dir}")

    # Run the generate-descriptions-for-entire-projects command
    try:
        run_generate_descriptions_for_entire_projects(
            base_project_dir,
            src_dir,
            output_dir,
            CLEAR_DATASET,
            MAX_ENTRIES,
            LLM_MODEL,
            ONLY_INTERESTING_TESTS,
        )
        print("Completed description generation...", flush=True)
    except subprocess.CalledProcessError:
        sys.exit(1)


if __name__ == "__main__":
    main()
