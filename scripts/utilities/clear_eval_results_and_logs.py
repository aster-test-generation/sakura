from pathlib import Path

# === CONFIGURATION CONSTANTS ===
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
EVAL_OUTPUT_DIR = "resources/output/"
EVAL_FILE_NAME = "nl2test_evaluation_results.json"
LOG_FILE_NAME = "nl2test.log"


def clear_eval_results_and_logs(output_dir: Path) -> None:
    """Remove evaluation results and logs from project output folders."""
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Evaluation output directory not found: {output_dir}")

    for project_dir in output_dir.iterdir():
        if not project_dir.is_dir():
            continue

        eval_file = project_dir / EVAL_FILE_NAME
        log_file = project_dir / LOG_FILE_NAME

        if eval_file.is_file():
            eval_file.unlink()
        if log_file.is_file():
            log_file.unlink()


def main() -> None:
    output_dir = (ROOT_DIR / EVAL_OUTPUT_DIR).resolve()
    clear_eval_results_and_logs(output_dir)


if __name__ == "__main__":
    main()
