"""Aggregate localization recall statistics from NL2Test evaluation results."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sakura.nl2test.models.decomposition import LocalizationEval
from sakura.utils.statistics import DistributionSummary, summarize_distribution

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_OUTPUTS_DIR = ROOT_DIR / "outputs" / "raw_outputs"
OUTPUT_DIR = ROOT_DIR / "outputs" / "localization_stats"
EVAL_FILE_NAME = "nl2test_evaluation_results.json"

NL2TEST_OUTPUT_DIRS = [
    RAW_OUTPUTS_DIR / "nl2test_devstral_output",
    RAW_OUTPUTS_DIR / "nl2test_gemini_flash_output",
    RAW_OUTPUTS_DIR / "nl2test_gemini_pro_output",
    RAW_OUTPUTS_DIR / "nl2test_qwen3_output",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate localization recall statistics from NL2Test results."
    )
    parser.add_argument(
        "--eval-file-name",
        type=str,
        default=EVAL_FILE_NAME,
        help="Evaluation file name inside each project directory.",
    )
    return parser.parse_args()


def _extract_localization_recall(entry: dict[str, Any]) -> float:
    """Extract localization recall from entry, returning 0.0 if missing or invalid."""
    loc_eval = entry.get("localization_eval")
    if loc_eval is None:
        return 0.0
    try:
        loc = LocalizationEval.model_validate(loc_eval)
        return loc.localization_recall
    except ValidationError:
        return 0.0


def _distribution_to_dict(dist: DistributionSummary) -> dict[str, Any]:
    return asdict(dist)


def collect_localization_stats(output_dir: Path, eval_file_name: str) -> dict[str, Any]:
    """Collect localization recall stats from all projects in output_dir."""
    project_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir()], key=lambda p: p.name
    )

    per_project_stats: dict[str, dict[str, Any]] = {}
    all_recalls: list[float] = []

    for project_dir in project_dirs:
        eval_file = project_dir / eval_file_name
        if not eval_file.exists():
            continue

        try:
            with eval_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue

        project_recalls = [_extract_localization_recall(entry) for entry in data]
        all_recalls.extend(project_recalls)

        per_project_stats[project_dir.name] = {
            "entries": len(project_recalls),
            "recall_distribution": _distribution_to_dict(
                summarize_distribution(project_recalls)
            ),
        }

    return {
        "summary": {
            "output_directory": str(output_dir.relative_to(ROOT_DIR)),
            "total_entries": len(all_recalls),
            "overall_recall_distribution": _distribution_to_dict(
                summarize_distribution(all_recalls)
            ),
        },
        "per_project": per_project_stats,
    }


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for output_dir in NL2TEST_OUTPUT_DIRS:
        if not output_dir.exists():
            print(f"Skipping {output_dir.name}: directory does not exist")
            continue

        print(f"Processing {output_dir.name}...")
        stats = collect_localization_stats(output_dir, args.eval_file_name)

        output_file = OUTPUT_DIR / f"{output_dir.name}_localization.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        summary = stats["summary"]
        recall = summary["overall_recall_distribution"]
        print(f"  Entries: {summary['total_entries']}")
        print(f"  Recall: mean={recall['mean']:.4f}, p50={recall['p50']:.4f}")
        print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()
