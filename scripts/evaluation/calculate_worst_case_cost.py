"""Calculate worst-case per-input cost for NL2Test models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_OUTPUTS_DIR = ROOT_DIR / "outputs" / "raw_outputs"
EVAL_FILE_NAME = "nl2test_evaluation_results.json"

NL2TEST_DIRS = {
    "nl2test_gemini_flash_output": "gemini-2.5-flash",
    "nl2test_gemini_pro_output": "gemini-2.5-pro",
    "nl2test_devstral_output": "devstral-small-latest",
    "nl2test_qwen3_output": "qwen/qwen3-coder",
}

MODEL_PRICING = {
    "gemini-2.5-flash": {
        "input_per_million": 0.30,
        "output_per_million": 2.50,
    },
    "gemini-2.5-pro": {
        "input_per_million_under_200k": 1.25,
        "input_per_million_over_200k": 2.50,
        "output_per_million_under_200k": 10.00,
        "output_per_million_over_200k": 15.00,
        "context_threshold": 200_000,
    },
    "devstral-small-latest": {
        "input_per_million": 0.1,
        "output_per_million": 0.3,
    },
    "qwen/qwen3-coder": {
        "input_per_million": 0.22,
        "output_per_million": 0.95,
    },
}


def calculate_entry_cost(
    input_tokens: float, output_tokens: float, pricing_model: str
) -> float:
    """Calculate cost for a single entry based on token counts and pricing model."""
    pricing = MODEL_PRICING[pricing_model]

    if "input_per_million" in pricing and "output_per_million" in pricing:
        input_cost = (input_tokens / 1_000_000) * pricing["input_per_million"]
        output_cost = (output_tokens / 1_000_000) * pricing["output_per_million"]
    else:
        threshold = pricing["context_threshold"]
        if input_tokens <= threshold:
            input_cost = (input_tokens / 1_000_000) * pricing[
                "input_per_million_under_200k"
            ]
            output_cost = (output_tokens / 1_000_000) * pricing[
                "output_per_million_under_200k"
            ]
        else:
            input_cost = (input_tokens / 1_000_000) * pricing[
                "input_per_million_over_200k"
            ]
            output_cost = (output_tokens / 1_000_000) * pricing[
                "output_per_million_over_200k"
            ]

    return input_cost + output_cost


def load_eval_entries(eval_file: Path) -> list[dict[str, Any]]:
    """Load evaluation entries from a JSON file."""
    try:
        with eval_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, IOError):
        pass
    return []


def main() -> None:
    print("=" * 70)
    print("Worst-Case Per-Input Cost Analysis for NL2Test Models")
    print("=" * 70)

    for dir_name, pricing_model in NL2TEST_DIRS.items():
        output_dir = RAW_OUTPUTS_DIR / dir_name
        if not output_dir.exists():
            print(f"\n{dir_name}: Directory not found")
            continue

        max_cost = 0.0
        max_input_tokens = 0
        max_output_tokens = 0
        max_entry_id = None
        max_project = None
        total_entries = 0

        project_dirs = [p for p in output_dir.iterdir() if p.is_dir()]
        for project_dir in project_dirs:
            eval_file = project_dir / EVAL_FILE_NAME
            if not eval_file.is_file():
                continue

            entries = load_eval_entries(eval_file)
            for entry in entries:
                input_tokens = entry.get("input_tokens", 0)
                output_tokens = entry.get("output_tokens", 0)
                entry_id = entry.get("nl2test_input", {}).get("id", "unknown")

                cost = calculate_entry_cost(input_tokens, output_tokens, pricing_model)
                total_entries += 1

                if cost > max_cost:
                    max_cost = cost
                    max_input_tokens = input_tokens
                    max_output_tokens = output_tokens
                    max_entry_id = entry_id
                    max_project = project_dir.name

        print(f"\n{dir_name}")
        print("-" * len(dir_name))
        print(f"  Pricing model:     {pricing_model}")
        print(f"  Total entries:     {total_entries}")
        print(f"  Worst-case cost:   ${max_cost:.4f}")
        print(f"  Max input tokens:  {max_input_tokens:,}")
        print(f"  Max output tokens: {max_output_tokens:,}")
        print(f"  Entry ID:          {max_entry_id}")
        print(f"  Project:           {max_project}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
