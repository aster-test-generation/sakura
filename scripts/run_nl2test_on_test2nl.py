import sys
import subprocess
from pathlib import Path

# === CONFIGURATION CONSTANTS ===
# Directory paths relative to this script
SRC_DIR = "../src"
BASE_PROJECT_DIR = "../tests/resources"
BASE_ANALYSIS_DIR = "../tests/output/"
OUTPUT_DIR = "../tests/output"
# Test2NL CSV file path (must include the filename)
CSV_FILE = "../tests/output/resources/test2nl/test2nl.csv"

# CLI arguments
MAX_ENTRIES = 0
# Note: 0 = unlimited
CLEAR_OUTPUT = True

LLM_MODEL = "google/gemini-2.5-flash"
# Either LLM_PROVIDER or LLM_API_URL must be non-None
LLM_PROVIDER: str | None = "openrouter"  # Supported providers: "openrouter", "ollama", "vllm", "openai", "gcp"
LLM_API_URL: str | None = None  # OpenAI-compatible base URL if overriding

EMB_MODEL = "nomic-embed-text:v1.5"
# Either EMB_PROVIDER or EMB_API_URL must be non-None
EMB_PROVIDER: str | None = "ollama"  # Supported providers: "ollama", "openrouter", "vllm", "openai", "gcp"
EMB_API_URL: str | None = None

# Decomposition mode
DECOMPOSITION_MODE = "gherkin"  # Only supporting "gherkin" atm.

# Iteration settings on agents (trajectory length ceiling)
SUPERVISOR_MAX_ITERS: int = 10
LOCALIZATION_MAX_ITERS: int = 40
COMPOSITION_MAX_ITERS: int = 30

# Parallelization defaults (only between projects)
NUM_PROJ_PARALLEL = 2
MAX_INFLIGHT = 0  # 0 uses num_proj_parallel


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    # Set up paths using configuration constants
    src_dir = (script_dir / SRC_DIR).resolve()
    base_project_dir = (script_dir / BASE_PROJECT_DIR).resolve()
    output_dir = (script_dir / OUTPUT_DIR).resolve()
    test2nl_file = (script_dir / CSV_FILE).resolve()
    base_analysis_dir = (
        (script_dir / BASE_ANALYSIS_DIR).resolve() if BASE_ANALYSIS_DIR else None
    )

    # Verify paths exist
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    cli_file = src_dir / "nltest" / "cli.py"
    if not cli_file.is_file():
        raise FileNotFoundError(f"CLI not found at expected path: {cli_file}")

    if not base_project_dir.is_dir():
        raise FileNotFoundError(f"Base project directory not found: {base_project_dir}")

    if base_analysis_dir is None or not base_analysis_dir.is_dir():
        raise FileNotFoundError(
            "Base analysis directory not set or not found. "
            "Please set BASE_ANALYSIS_DIR to the path containing per-project analysis.json folders."
        )

    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    if not test2nl_file.is_file():
        raise FileNotFoundError(f"Test2NL CSV not found: {test2nl_file}")

    # Construct the command
    cmd = [
        "poetry",
        "run",
        "python",
        "-u",
        "-m",
        "nltest.cli",
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
        "--emb-model",
        EMB_MODEL,
        "--decomposition-mode",
        DECOMPOSITION_MODE,
        "--num-proj-parallel",
        str(NUM_PROJ_PARALLEL),
        "--max-inflight",
        str(MAX_INFLIGHT),
    ]

    # Optional connectivity flags
    if LLM_PROVIDER:
        cmd.extend(["--llm-provider", LLM_PROVIDER])
    if LLM_API_URL:
        cmd.extend(["--llm-api-url", LLM_API_URL])
    if EMB_PROVIDER:
        cmd.extend(["--emb-provider", EMB_PROVIDER])
    if EMB_API_URL:
        cmd.extend(["--emb-api-url", EMB_API_URL])

    if CLEAR_OUTPUT:
        cmd.append("--clear-output")

    # Optional caps
    if MAX_ENTRIES and MAX_ENTRIES > 0:
        cmd.extend(["--max-entries", str(MAX_ENTRIES)])

    # Optional iteration overrides
    if SUPERVISOR_MAX_ITERS is not None:
        cmd.extend(["--supervisor-max-iters", str(SUPERVISOR_MAX_ITERS)])
    if LOCALIZATION_MAX_ITERS is not None:
        cmd.extend(["--localization-max-iters", str(LOCALIZATION_MAX_ITERS)])
    if COMPOSITION_MAX_ITERS is not None:
        cmd.extend(["--composition-max-iters", str(COMPOSITION_MAX_ITERS)])

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
