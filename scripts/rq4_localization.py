import os
import sys
import subprocess
from pathlib import Path

def main() -> None:
    script_dir = Path(__file__).resolve().parent
    
    # Set up paths relative to the script
    src_dir = (script_dir / ".." / "src").resolve()
    base_project_dir = (script_dir / ".." / "tests" / "resources").resolve()
    output_dir = (script_dir / ".." / "tests"/ "output").resolve()
    csv_file = "spring-petclinic/test2nl.csv"
    
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
        "-m",
        "nltest.cli",
        "evaluate-localization",
        "--base-project-dir",
        str(base_project_dir),
        "--output-dir",
        str(output_dir),
        "--csv-file",
        csv_file,
    ]
    
    print(f"Running RQ4 localization evaluation...", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=src_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("RQ4 localization evaluation completed successfully!", flush=True)
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"RQ4 localization evaluation failed!", flush=True)
        print(f"Return code: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    main()
