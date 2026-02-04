"""Analyze tool call statistics from evaluation outputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from sakura.utils.models.nl2test import NL2TestEval, ToolLog
from sakura.utils.statistics import (
    distribution_to_dict,
    summarize_distribution,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_OUTPUTS_DIR = ROOT_DIR / "outputs" / "raw_outputs"
OUTPUT_DIR = ROOT_DIR / "outputs" / "tool_stats"

EVAL_FILE_NAME = "nl2test_evaluation_results.json"

EVAL_DIRS: List[Path] = [
    RAW_OUTPUTS_DIR / "gemini_cli_pro_output",
    RAW_OUTPUTS_DIR / "gemini_cli_flash_output",
    RAW_OUTPUTS_DIR / "nl2test_gemini_pro_output",
    RAW_OUTPUTS_DIR / "nl2test_gemini_flash_output",
    RAW_OUTPUTS_DIR / "nl2test_devstral_output",
    RAW_OUTPUTS_DIR / "nl2test_qwen3_output",
]


def load_tool_data(output_dir: Path) -> tuple[List[Optional[ToolLog]], int, int]:
    """Load tool_log from evaluation results.

    Returns:
        Tuple of (list of ToolLog objects, total entries, entries with tool data)
    """
    entries: List[Optional[ToolLog]] = []
    total_entries = 0
    entries_with_data = 0

    for project_dir in output_dir.iterdir():
        if not project_dir.is_dir():
            continue
        eval_file = project_dir / EVAL_FILE_NAME
        if not eval_file.is_file():
            continue
        try:
            with eval_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue

        if not isinstance(data, list):
            continue

        for item in data:
            total_entries += 1
            try:
                eval_entry = NL2TestEval.model_validate(item)
                if eval_entry.tool_log is not None:
                    entries.append(eval_entry.tool_log)
                    entries_with_data += 1
                else:
                    entries.append(None)
            except ValidationError:
                entries.append(None)

    return entries, total_entries, entries_with_data


def aggregate_tool_stats(tool_logs: List[ToolLog]) -> Dict[str, Any]:
    """Aggregate tool statistics from tool logs."""
    total_calls = 0
    all_tools: set[str] = set()
    calls_per_entry: List[float] = []

    by_agent: Dict[str, Dict[str, Any]] = {
        "supervisor": {"total_calls": 0, "tools": defaultdict(list)},
        "localization": {"total_calls": 0, "tools": defaultdict(list)},
        "composition": {"total_calls": 0, "tools": defaultdict(list)},
    }
    by_tool: Dict[str, List[int]] = defaultdict(list)

    for tool_log in tool_logs:
        entry_total = 0
        agent_logs = [
            ("supervisor", tool_log.supervisor_tool_log),
            ("localization", tool_log.localization_tool_log),
            ("composition", tool_log.composition_tool_log),
        ]
        for agent_name, agent_log in agent_logs:
            for tool_name, count in agent_log.tool_counts.items():
                entry_total += count
                by_agent[agent_name]["total_calls"] += count
                by_agent[agent_name]["tools"][tool_name].append(count)
                by_tool[tool_name].append(count)
                all_tools.add(tool_name)

        total_calls += entry_total
        calls_per_entry.append(float(entry_total))

    overall = {
        "total_tool_calls": total_calls,
        "unique_tools": len(all_tools),
        "calls_per_entry": distribution_to_dict(
            summarize_distribution(calls_per_entry)
        ),
    }

    by_agent_stats: Dict[str, Any] = {}
    for agent_name, agent_data in by_agent.items():
        tool_stats: Dict[str, Any] = {}
        for tool_name, counts in sorted(agent_data["tools"].items()):
            total_for_tool = sum(counts)
            tool_stats[tool_name] = {
                "total": total_for_tool,
                "entries_using": len(counts),
                "distribution": distribution_to_dict(
                    summarize_distribution([float(c) for c in counts])
                ),
            }
        by_agent_stats[agent_name] = {
            "total_calls": agent_data["total_calls"],
            "tools": tool_stats,
        }

    by_tool_stats: Dict[str, Any] = {}
    for tool_name in sorted(all_tools):
        counts = by_tool[tool_name]
        total_for_tool = sum(counts)
        by_tool_stats[tool_name] = {
            "total": total_for_tool,
            "entries_using": len(counts),
            "distribution": distribution_to_dict(
                summarize_distribution([float(c) for c in counts])
            ),
        }

    return {"overall": overall, "by_agent": by_agent_stats, "by_tool": by_tool_stats}


def print_tool_stats(stats: Dict[str, Any], dir_name: str) -> None:
    """Print tool statistics to console."""
    print(f"\n{'=' * 60}")
    print(f"Tool Statistics: {dir_name}")
    print("=" * 60)

    overall = stats["overall"]
    print(f"\nTotal tool calls: {overall['total_tool_calls']}")
    print(f"Unique tools: {overall['unique_tools']}")
    dist = overall["calls_per_entry"]
    print(f"Calls per entry: mean={dist['mean']:.2f}, std={dist['std']:.2f}")

    print("\nBy Agent:")
    for agent_name, agent_data in stats["by_agent"].items():
        print(f"\n  {agent_name.upper()}: {agent_data['total_calls']} total calls")
        if agent_data["tools"]:
            print(f"    {'Tool Name':<28} {'Total':>8} {'Entries':>8} {'Mean':>8}")
            print(f"    {'-' * 28} {'-' * 8} {'-' * 8} {'-' * 8}")
            for tool_name, tool_data in agent_data["tools"].items():
                mean = tool_data["distribution"]["mean"]
                print(
                    f"    {tool_name:<28} {tool_data['total']:>8} "
                    f"{tool_data['entries_using']:>8} {mean:>8.2f}"
                )

    print("\nBy Tool (All Agents):")
    print(f"  {'Tool Name':<30} {'Total':>8} {'Entries':>8} {'Mean':>8}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8}")
    for tool_name, tool_data in stats["by_tool"].items():
        mean = tool_data["distribution"]["mean"]
        print(
            f"  {tool_name:<30} {tool_data['total']:>8} "
            f"{tool_data['entries_using']:>8} {mean:>8.2f}"
        )


def evaluate_directory(output_dir: Path) -> None:
    """Process one directory and generate tool statistics."""
    if not output_dir.exists():
        print(f"Directory not found: {output_dir}")
        return

    dir_name = output_dir.name
    print(f"\nProcessing: {output_dir}")

    tool_logs, total_entries, entries_with_data = load_tool_data(output_dir)
    valid_logs = [log for log in tool_logs if log is not None]
    stats = aggregate_tool_stats(valid_logs)
    print_tool_stats(stats, dir_name)

    json_output: Dict[str, Any] = {
        "summary": {
            "output_directory": str(output_dir.relative_to(ROOT_DIR)),
            "total_entries": total_entries,
            "entries_with_tool_data": entries_with_data,
        },
        **stats,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_name = f"{dir_name}_tool_stats.json"
    output_file = OUTPUT_DIR / output_name
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2)
    print(f"\nJSON results written to: {output_file}")


def main() -> None:
    for eval_dir in EVAL_DIRS:
        evaluate_directory(eval_dir)


if __name__ == "__main__":
    main()
