import subprocess
import sys
from pathlib import Path

# === CONFIGURATION CONSTANTS ===
# Root directory (project root, two levels up from this script)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Directory paths relative to ROOT_DIR
SRC_DIR = "src"
BASE_PROJECT_DIR = "resources/datasets/"
BASE_ANALYSIS_DIR = "resources/analysis/"
OUTPUT_DIR = "outputs/raw_outputs/OUTPUT_DIR"
# Log file name to write under OUTPUT_DIR
LOG_FILE_NAME = "nl2test.log"  # Only the name; saved inside OUTPUT_DIR
# Test2NL CSV file path (must include the filename)
CSV_FILE = "resources/test2nl/filtered_dataset/test2nl.csv"

# CLI arguments
MAX_ENTRIES = 0
# Note: 0 = unlimited
TARGET_IDS: list[int] = []  # Empty list processes all; non-empty filters to these IDs
DEBUG = True
USE_STORED_INDEX = True
RESET_EVALUATION_RESULTS = True
EXCLUDE_TEST_DIRS = True

LLM_MODEL = "google/gemini-2.5-flash"
# Either LLM_PROVIDER or LLM_API_URL must be non-None
LLM_PROVIDER: str | None = (
    "openrouter"  # Supported providers: "openrouter", "ollama", "vllm", "openai", "gcp", "mistral"
)
LLM_API_URL: str | None = None  # OpenAI-compatible base URL if overriding

EMB_MODEL = "qwen/qwen3-embedding-8b"
# Either EMB_PROVIDER or EMB_API_URL must be non-None
EMB_PROVIDER: str | None = (
    "openrouter"  # Supported providers: "ollama", "openrouter", "vllm", "openai", "gcp"
)
EMB_API_URL: str | None = None

# Decomposition mode
DECOMPOSITION_MODE = "gherkin"  # Only supporting "gherkin" atm.

# Parallel tool call behavior for the LLM
CAN_PARALLEL_TOOL_CALL: bool = True

# Config reasoning abilities (only applies when using OpenRouter provider)
CONFIGURE_REASONING: bool = False
REASONING_EFFORT: str = "minimal"
EXCLUDE_REASONING: bool = False

# Provider ignore list for OpenRouter (only applies when using OpenRouter provider)
OPENROUTER_IGNORE_PROVIDERS: list[str] | None = None # e.g., ["nebius/fp8"]
# Note: "nebius/fp8" has problems when using qwen3-coder

# Maximum tokens for LLM completion output.
# Default: 16384. Use 32768 or 65536 for extensive reasoning models like MiniMax M2.1.
MAX_TOKENS: int = 16384

# Iteration settings on agents (trajectory length ceiling)
SUPERVISOR_MAX_ITERS: int = 8
LOCALIZATION_MAX_ITERS: int = 12
COMPOSITION_MAX_ITERS: int = 14

# Parallelization defaults (only between projects)
NUM_PROJ_PARALLEL = 10
MAX_INFLIGHT = 0  # 0 uses num_proj_parallel

# Save localized scenarios to separate JSON file per-project (GHERKIN mode only)
SAVE_LOCALIZED_SCENARIOS: bool = False

# Save each generate_test_code iteration to code_iteration/ directory for debugging
STORE_CODE_ITERATION: bool = False


def main() -> None:
    # Set up paths using configuration constants
    src_dir = (ROOT_DIR / SRC_DIR).resolve()
    base_project_dir = (ROOT_DIR / BASE_PROJECT_DIR).resolve()
    output_dir = (ROOT_DIR / OUTPUT_DIR).resolve()
    log_file_name = LOG_FILE_NAME  # Name only; CLI saves under output_dir
    test2nl_file = (ROOT_DIR / CSV_FILE).resolve()
    base_analysis_dir = (
        (ROOT_DIR / BASE_ANALYSIS_DIR).resolve() if BASE_ANALYSIS_DIR else None
    )

    # Verify paths exist
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    cli_file = src_dir / "sakura" / "cli.py"
    if not cli_file.is_file():
        raise FileNotFoundError(f"CLI not found at expected path: {cli_file}")

    if not base_project_dir.is_dir():
        raise FileNotFoundError(f"Base project directory not found: {base_project_dir}")

    if base_analysis_dir is None or not base_analysis_dir.is_dir():
        raise FileNotFoundError(
            "Base analysis directory not set or not found. "
            "Please set BASE_ANALYSIS_DIR to the path containing per-project analysis.json folders."
        )

    if output_dir.exists() and not output_dir.is_dir():
        raise FileNotFoundError(f"Output path is not a directory: {output_dir}")
    if not output_dir.exists():
        # Create output directory on demand for generated artifacts.
        output_dir.mkdir(parents=True, exist_ok=True)

    if not test2nl_file.is_file():
        raise FileNotFoundError(f"Test2NL CSV not found: {test2nl_file}")

    # Construct the command
    cmd = [
        "poetry",
        "run",
        "python",
        "-u",
        "-m",
        "sakura.cli",
        "run-nl2test",
        "--base-project-dir",
        str(base_project_dir),
        "--base-analysis-dir",
        str(base_analysis_dir),
        "--output-dir",
        str(output_dir),
        "--test2nl-file",
        str(test2nl_file),
        "--llm-model",
        LLM_MODEL,
        "--can-parallel-tool" if CAN_PARALLEL_TOOL_CALL else "--no-can-parallel-tool",
        "--emb-model",
        EMB_MODEL,
        "--decomposition-mode",
        DECOMPOSITION_MODE,
        "--num-proj-parallel",
        str(NUM_PROJ_PARALLEL),
        "--max-inflight",
        str(MAX_INFLIGHT),
    ]

    if EXCLUDE_TEST_DIRS:
        cmd.append("--exclude-test-dirs")
    else:
        cmd.append("--no-exclude-test-dirs")

    # Optional connectivity flags
    if LLM_PROVIDER:
        cmd.extend(["--llm-provider", LLM_PROVIDER])
    if LLM_API_URL:
        cmd.extend(["--llm-api-url", LLM_API_URL])
    if EMB_PROVIDER:
        cmd.extend(["--emb-provider", EMB_PROVIDER])
    if EMB_API_URL:
        cmd.extend(["--emb-api-url", EMB_API_URL])

    if RESET_EVALUATION_RESULTS:
        cmd.append("--reset-evaluation-results")
    else:
        cmd.append("--no-reset-evaluation-results")

    if USE_STORED_INDEX:
        cmd.append("--use-stored-index")
    else:
        cmd.append("--no-use-stored-index")
    if DEBUG:
        cmd.append("--debug")
    if log_file_name:
        cmd.extend(["--log-file", log_file_name])

    # Optional caps
    if MAX_ENTRIES and MAX_ENTRIES > 0:
        cmd.extend(["--max-entries", str(MAX_ENTRIES)])
    if TARGET_IDS:
        for target_id in TARGET_IDS:
            cmd.extend(["--target-ids", str(target_id)])

    # Optional iteration overrides
    if SUPERVISOR_MAX_ITERS is not None:
        cmd.extend(["--supervisor-max-iters", str(SUPERVISOR_MAX_ITERS)])
    if LOCALIZATION_MAX_ITERS is not None:
        cmd.extend(["--localization-max-iters", str(LOCALIZATION_MAX_ITERS)])
    if COMPOSITION_MAX_ITERS is not None:
        cmd.extend(["--composition-max-iters", str(COMPOSITION_MAX_ITERS)])

    if CONFIGURE_REASONING:
        cmd.append("--configure-reasoning")
    else:
        cmd.append("--no-configure-reasoning")
    cmd.extend(["--reasoning-effort", REASONING_EFFORT])
    if EXCLUDE_REASONING:
        cmd.append("--exclude-reasoning")
    else:
        cmd.append("--no-exclude-reasoning")
    cmd.extend(["--max-tokens", str(MAX_TOKENS)])

    if OPENROUTER_IGNORE_PROVIDERS:
        for provider_name in OPENROUTER_IGNORE_PROVIDERS:
            cmd.extend(["--openrouter-ignore-providers", provider_name])

    if SAVE_LOCALIZED_SCENARIOS:
        cmd.append("--save-localized-scenarios")

    if STORE_CODE_ITERATION:
        cmd.append("--store-code-iteration")

    print("Running NL2Test on Test2NL inputs...", flush=True)
    print(f"Decomposition mode: {DECOMPOSITION_MODE}", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)

    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=src_dir,
        )
        print("NL2Test run completed successfully!", flush=True)
    except subprocess.CalledProcessError as e:
        print("NL2Test run failed!", flush=True)
        print(f"Return code: {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
