import os
import sys
import subprocess
from pathlib import Path

# === CONFIGURATION CONSTANTS ===
# Directory paths relative to this script
SRC_DIR = "../src"
BASE_PROJECT_DIR = "../tests/resources"
OUTPUT_DIR = "../tests/output"

# CLI arguments
CSV_FILE = "spring-petclinic/test2nl.csv"
MAX_ENTRIES = 6  # Note: 0 = unlimited
LLM_MODEL = "mistralai/devstral-small"
DECOMPOSITION_MODE = "grammatical"  # or "gherkin"

def main() -> None:
    script_dir = Path(__file__).resolve().parent
    
    # Set up paths using configuration constants
    src_dir = (script_dir / SRC_DIR).resolve()
    base_project_dir = (script_dir / BASE_PROJECT_DIR).resolve()
    output_dir = (script_dir / OUTPUT_DIR).resolve()
    csv_file = CSV_FILE
    
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
        "--csv-file",
        csv_file,
        "--llm-model",
        LLM_MODEL,
    ]
    
    if MAX_ENTRIES > 0:
        cmd.extend(["--max-entries", str(MAX_ENTRIES)])

    if DECOMPOSITION_MODE:
        cmd.extend(["--decomposition-mode", DECOMPOSITION_MODE])
    
    print(f"Running RQ4 localization evaluation...", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=src_dir,
        )
        print("RQ4 localization evaluation completed successfully!", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"RQ4 localization evaluation failed!", flush=True)
        print(f"Return code: {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
