import sys
import subprocess
from pathlib import Path

# === CONFIGURATION CONSTANTS ===
# Directory paths relative to this script
SRC_DIR = "../src"
BASE_PROJECT_DIR = "../tests/resources"
OUTPUT_DIR = "../tests/output"
CSV_FILE = "../tests/output/resources/test2nl/test2nl.csv"

# CLI arguments
MAX_ENTRIES = 0  # Note: 0 = unlimited
LLM_MODEL = "google/gemini-2.5-flash"

# Localization settings
LOCALIZATION_MAX_ITERS = 50

# Decomposition mode (must match DecompositionMode enum values)
_ALLOWED_DECOMP_MODES = {"grammatical", "gherkin"}
DECOMPOSITION_MODE = "gherkin"
if DECOMPOSITION_MODE not in _ALLOWED_DECOMP_MODES:
    raise ValueError(
        f"Invalid DECOMPOSITION_MODE='{DECOMPOSITION_MODE}'. Choose one of {_ALLOWED_DECOMP_MODES}."
    )

# Parallelization defaults
NUM_PROJ_PARALLEL = 2
PER_PROJ_CONCURRENCY = 10
MAX_INFLIGHT = 0  # 0 => unbounded (uses num_proj_parallel * per_proj_concurrency)


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    # Set up paths using configuration constants
    src_dir = (script_dir / SRC_DIR).resolve()
    base_project_dir = (script_dir / BASE_PROJECT_DIR).resolve()
    output_dir = (script_dir / OUTPUT_DIR).resolve()
    test2nl_file = (script_dir / CSV_FILE).resolve()

    # Verify paths exist
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    cli_file = src_dir / "nltest" / "cli.py"
    if not cli_file.is_file():
        raise FileNotFoundError(f"CLI not found at expected path: {cli_file}")

    if not base_project_dir.is_dir():
        raise FileNotFoundError(f"Base project directory not found: {base_project_dir}")

    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    # Construct the command
    cmd = [
        "poetry",
        "run",
        "python",
        "-u",
        "-m",
        "nltest.cli",
        "evaluate-localization",
        "--base-project-dir",
        str(base_project_dir),
        "--output-dir",
        str(output_dir),
        "--test2nl-file",
        str(test2nl_file),
        "--llm-model",
        LLM_MODEL,
        "--decomposition-mode",
        DECOMPOSITION_MODE,
        "--localization-max-iters",
        str(LOCALIZATION_MAX_ITERS),
        "--num-proj-parallel",
        str(NUM_PROJ_PARALLEL),
        "--per-proj-concurrency",
        str(PER_PROJ_CONCURRENCY),
        "--max-inflight",
        str(MAX_INFLIGHT),
    ]

    if MAX_ENTRIES > 0:
        cmd.extend(["--max-entries", str(MAX_ENTRIES)])

    print("Running RQ4 localization evaluation...", flush=True)
    print(f"Decomposition mode: {DECOMPOSITION_MODE}", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)

    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=src_dir,
        )
        print("RQ4 localization evaluation completed successfully!", flush=True)
    except subprocess.CalledProcessError as e:
        print("RQ4 localization evaluation failed!", flush=True)
        print(f"Return code: {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
