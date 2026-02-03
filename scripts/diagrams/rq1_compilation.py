import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from colors import DIAGRAM_STATS_DIR, INPUT_FILES, OUTPUT_DIR, STATS_DIR, get_color

DISABLE_LEGEND = True


def load_results(file_path: Path) -> dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def save_compilation_stats(
    stats_dir: Path = STATS_DIR,
    diagram_stats_dir: Path = DIAGRAM_STATS_DIR,
) -> None:
    """Save compilation statistics to JSON for paper writing."""
    diagram_stats_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {}
    for filename, display_name in INPUT_FILES.items():
        file_path = stats_dir / filename
        if not file_path.exists():
            continue

        data = load_results(file_path)
        compile_rate = data["holistic"]["compile_rate"] * 100

        stats[display_name] = {
            "compile_rate_pct": round(compile_rate, 2),
        }

    output_path = diagram_stats_dir / "rq1_compilation_stats.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved compilation stats to {output_path}")


def create_compilation_bar_graph(
    stats_dir: Path = STATS_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Create bar graph of compilation rates for all models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names: list[str] = []
    compile_rates: list[float] = []
    colors: list[str] = []

    for filename, display_name in INPUT_FILES.items():
        file_path = stats_dir / filename
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping")
            continue

        data = load_results(file_path)
        compile_rate = data["holistic"]["compile_rate"] * 100

        model_names.append(display_name)
        compile_rates.append(compile_rate)
        colors.append(get_color(filename))

    fig, ax = plt.subplots(figsize=(5, 8))

    bar_width = 0.25
    x_positions = [i * bar_width for i in range(len(model_names))]
    bars = ax.bar(
        x_positions,
        compile_rates,
        width=bar_width,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_ylim(0, 100)
    ax.margins(x=0.15)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=28)

    if not DISABLE_LEGEND:
        ax.legend(bars, model_names, loc="upper left", fontsize=22)

    plt.tight_layout()

    output_path = output_dir / "rq1_compilation.pdf"
    plt.savefig(output_path)
    plt.close()

    print(f"Saved compilation bar graph to {output_path}")


def main() -> None:
    """Entry point."""
    create_compilation_bar_graph()
    save_compilation_stats()


if __name__ == "__main__":
    main()
