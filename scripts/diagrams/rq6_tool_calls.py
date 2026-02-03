import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from colors import DIAGRAM_STATS_DIR, INPUT_FILES, OUTPUT_DIR, ROOT_DIR, get_color

TOOL_ANALYSIS_DIR = ROOT_DIR / "outputs" / "tool_analysis"

CATEGORY_ORDER = [
    "retrieval",
    "inspection",
    "generation",
    "validation",
    "orchestration",
]

CATEGORY_LABELS = [
    "Retrieval",
    "Inspection",
    "Generation",
    "Validation",
    "Orchestration",
]

DISABLE_LEGEND = True


def load_results(file_path: Path) -> dict[str, Any]:
    """Load tool category analysis results from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def get_category_percentages(data: dict[str, Any]) -> list[float]:
    """Return category proportions as percentages in CATEGORY_ORDER."""
    totals = data.get("category_totals", {})
    total_count = sum(item.get("count", 0) for item in totals.values())
    percentages: list[float] = []
    for category in CATEGORY_ORDER:
        info = totals.get(category, {})
        proportion = info.get("proportion")
        if proportion is None:
            count = info.get("count", 0)
            proportion = count / total_count if total_count > 0 else 0.0
        percentages.append(proportion * 100)
    return percentages


def create_tool_category_bar_graph(
    model_files: list[str],
    model_names: list[str],
    model_data: dict[str, list[float]],
    output_dir: Path,
) -> None:
    """Create grouped bar graph of tool category proportions."""
    num_categories = len(CATEGORY_LABELS)
    num_models = len(model_files)
    bar_width = 0.12
    group_width = num_models * bar_width + 0.15

    fig, ax = plt.subplots(figsize=(12, 8))

    all_values: list[float] = []
    bars_list: list[Any] = []
    for model_idx, eval_filename in enumerate(model_files):
        values = model_data[eval_filename]
        all_values.extend(values)
        x_positions = [
            group_idx * group_width + model_idx * bar_width
            for group_idx in range(num_categories)
        ]
        bars = ax.bar(
            x_positions,
            values,
            width=bar_width,
            color=get_color(eval_filename),
            edgecolor="black",
            linewidth=0.5,
        )
        bars_list.append(bars)

    max_value = max(all_values) if all_values else 100
    y_limit = min(100, max_value * 1.1)
    ax.set_ylim(0, y_limit)

    group_centers = [
        group_idx * group_width + (num_models - 1) * bar_width / 2
        for group_idx in range(num_categories)
    ]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(CATEGORY_LABELS, fontsize=28, rotation=15, ha="right")
    ax.tick_params(axis="y", labelsize=28)

    if not DISABLE_LEGEND:
        ax.legend(bars_list, model_names, loc="upper right", fontsize=22)

    plt.tight_layout()

    output_path = output_dir / "rq6_tool_calls.pdf"
    plt.savefig(output_path)
    plt.close()

    print(f"Saved tool category graph to {output_path}")


def create_tool_category_chart(
    tool_analysis_dir: Path = TOOL_ANALYSIS_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Create tool category chart for all models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_files: list[str] = []
    model_names: list[str] = []
    model_data: dict[str, list[float]] = {}

    for eval_filename in INPUT_FILES:
        analysis_filename = eval_filename.replace(
            "_eval.json", "_category_analysis.json"
        )
        file_path = tool_analysis_dir / analysis_filename
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping")
            continue

        data = load_results(file_path)
        model_data[eval_filename] = get_category_percentages(data)
        model_files.append(eval_filename)
        model_names.append(INPUT_FILES[eval_filename])

    if not model_files:
        print("No data files found")
        return

    create_tool_category_bar_graph(
        model_files,
        model_names,
        model_data,
        output_dir,
    )


def save_tool_category_stats(
    tool_analysis_dir: Path = TOOL_ANALYSIS_DIR,
    diagram_stats_dir: Path = DIAGRAM_STATS_DIR,
) -> None:
    """Save tool category proportions for paper writing."""
    diagram_stats_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {}
    for eval_filename, display_name in INPUT_FILES.items():
        analysis_filename = eval_filename.replace(
            "_eval.json", "_category_analysis.json"
        )
        file_path = tool_analysis_dir / analysis_filename
        if not file_path.exists():
            continue

        data = load_results(file_path)
        percentages = get_category_percentages(data)
        stats[display_name] = {
            label: round(value, 2)
            for label, value in zip(CATEGORY_LABELS, percentages)
        }

    output_path = diagram_stats_dir / "rq6_tool_calls_stats.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved tool category stats to {output_path}")


def main() -> None:
    """Entry point."""
    create_tool_category_chart()
    save_tool_category_stats()


if __name__ == "__main__":
    main()
