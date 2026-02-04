import subprocess
import sys
from pathlib import Path

# === CONFIGURATION CONSTANTS ===
# Root directory (project root, two levels up from this script)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Directory paths relative to ROOT_DIR
SRC_DIR = "src"
ANALYSIS_DIR = "tests/output/resources/output"
ORGANIZED_METHODS_DIR = "tests/output/resources/bucketed_tests"
OUTPUT_DIR = "tests/output/resources/test2nl"
ORGANIZED_METHODS_FILE_NAME = "nl2test.json"

# CLI arguments
LLM_MODEL = "x-ai/grok-4-fast"
# Either LLM_PROVIDER or LLM_API_URL must be non-None
LLM_PROVIDER: str | None = (
    "openrouter"  # Supported providers: "openrouter", "ollama", "gcp"
)
LLM_API_URL: str | None = None  # must be OpenAI API compatible

CLEAR_DATASET = True
MAX_METHODS = 0  # Note: 0 = unlimited

# Parallelization defaults
NUM_PROJ_PARALLEL = 2
PER_PROJ_CONCURRENCY = 12
MAX_INFLIGHT = (
    0  # 0 is unbounded and defaults to num_proj_parallel * per_proj_concurrency
)

# Exclude specific dataset groups by name
# Valid options include:
#   - "tests_with_one_focal_methods"
#   - "tests_with_two_focal_methods"
#   - "tests_with_more_than_two_to_five_focal_methods"
#   - "tests_with_more_than_five_to_ten_focal_methods"
#   - "tests_with_more_than_ten_focal_methods"
EXCLUDE_GROUPS: list[str] = []


def main() -> None:
    # Resolve paths relative to ROOT_DIR
    src_dir = (ROOT_DIR / SRC_DIR).resolve()
    analysis_dir = (ROOT_DIR / ANALYSIS_DIR).resolve()
    organized_methods_dir = (ROOT_DIR / ORGANIZED_METHODS_DIR).resolve()
    output_dir = (ROOT_DIR / OUTPUT_DIR).resolve()

    # Verify paths exist
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    cli_file = src_dir / "sakura" / "cli.py"
    if not cli_file.is_file():
        raise FileNotFoundError(f"CLI not found at expected path: {cli_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct the command
    cmd = [
        "poetry",
        "run",
        "python",
        "-u",
        "-m",
        "sakura.cli",
        "generate-descriptions",
        "--analysis-dir",
        str(analysis_dir),
        "--output-dir",
        str(output_dir),
        "--organized-methods-dir",
        str(organized_methods_dir),
        "--organized-methods-file-name",
        ORGANIZED_METHODS_FILE_NAME,
        "--llm-model",
        LLM_MODEL,
        "--num-proj-parallel",
        str(NUM_PROJ_PARALLEL),
        "--per-proj-concurrency",
        str(PER_PROJ_CONCURRENCY),
        "--max-inflight",
        str(MAX_INFLIGHT),
    ]

    # Optional connectivity flags (only if provided)
    if LLM_PROVIDER:
        cmd.extend(["--llm-provider", LLM_PROVIDER])
    if LLM_API_URL:
        cmd.extend(["--llm-api-url", LLM_API_URL])

    if CLEAR_DATASET:
        cmd.append("--clear-dataset")
    if MAX_METHODS > 0:
        cmd.extend(["--max-methods", str(MAX_METHODS)])
    # Add any group exclusions
    for grp in EXCLUDE_GROUPS:
        cmd.extend(["--exclude-groups", grp])

    print("Running filtered Test2NL description generation...", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)

    try:
        subprocess.run(cmd, check=True, cwd=src_dir)
        print("Filtered description generation completed successfully!", flush=True)
    except subprocess.CalledProcessError as e:
        print("Filtered description generation failed!", flush=True)
        print(f"Return code: {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
