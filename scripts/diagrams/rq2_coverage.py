import json
import math
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from colors import DIAGRAM_STATS_DIR, INPUT_FILES, OUTPUT_DIR, STATS_DIR, get_color

COVERAGE_TYPES = [
    "class_coverage",
    "method_coverage",
    "line_coverage",
    "branch_coverage",
]
COVERAGE_LABELS = ["Class", "Method", "Line", "Branch"]

GraphType = Literal["bar_graph", "box_plot"]
GRAPH_TYPE: GraphType = "bar_graph"
DISABLE_LEGEND = True


def load_results(file_path: Path) -> dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def calculate_95_ci(std: float, count: int) -> float:
    """Calculate 95% confidence interval half-width."""
    return 1.96 * (std / math.sqrt(count))


def create_coverage_bar_graph(
    model_data: dict[str, dict[str, dict[str, float]]],
    filenames: list[str],
    model_names: list[str],
    output_dir: Path,
) -> None:
    """Create grouped bar graph of coverage metrics with 95% CI error bars."""
    num_groups = len(COVERAGE_LABELS)
    num_models = len(filenames)
    bar_width = 0.12
    group_width = num_models * bar_width + 0.15

    fig, ax = plt.subplots(figsize=(12, 8))

    bars_list = []
    for model_idx, filename in enumerate(filenames):
        values = [
            model_data[filename][cov_type]["mean"] * 100 for cov_type in COVERAGE_TYPES
        ]

        # Calculate 95% CI error bars
        ci_errors = [
            calculate_95_ci(
                model_data[filename][cov_type]["std"],
                model_data[filename][cov_type]["count"],
            )
            * 100
            for cov_type in COVERAGE_TYPES
        ]

        x_positions = [
            group_idx * group_width + model_idx * bar_width
            for group_idx in range(num_groups)
        ]
        bars = ax.bar(
            x_positions,
            values,
            width=bar_width,
            color=get_color(filename),
            edgecolor="black",
            linewidth=0.5,
            yerr=ci_errors,
            capsize=2,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
        bars_list.append(bars)

    ax.set_ylim(0, 100)

    group_centers = [
        group_idx * group_width + (num_models - 1) * bar_width / 2
        for group_idx in range(num_groups)
    ]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(COVERAGE_LABELS, fontsize=28)
    ax.tick_params(axis="y", labelsize=28)

    if not DISABLE_LEGEND:
        ax.legend(bars_list, model_names, loc="upper right", fontsize=22)

    plt.tight_layout()

    output_path = output_dir / "rq2_coverage.pdf"
    plt.savefig(output_path)
    plt.close()

    print(f"Saved coverage bar graph to {output_path}")


def create_coverage_box_plot(
    model_data: dict[str, dict[str, dict[str, float]]],
    filenames: list[str],
    model_names: list[str],
    output_dir: Path,
) -> None:
    """Create grouped box plot of coverage metrics."""
    num_groups = len(COVERAGE_LABELS)
    num_models = len(filenames)
    box_width = 0.12
    group_width = num_models * box_width + 0.15

    fig, ax = plt.subplots(figsize=(12, 8))

    for model_idx, filename in enumerate(filenames):
        positions = [
            group_idx * group_width + model_idx * box_width
            for group_idx in range(num_groups)
        ]

        for group_idx, cov_type in enumerate(COVERAGE_TYPES):
            stats = model_data[filename][cov_type]

            # Scale to percentage (using p10-p90 for whiskers)
            box_stats = {
                "med": stats["p50"] * 100,
                "q1": stats["p25"] * 100,
                "q3": stats["p75"] * 100,
                "whislo": stats["p10"] * 100,
                "whishi": stats["p90"] * 100,
            }

            bp = ax.bxp(
                [box_stats],
                positions=[positions[group_idx]],
                widths=box_width * 0.8,
                patch_artist=True,
                showfliers=False,
            )

            for patch in bp["boxes"]:
                patch.set_facecolor(get_color(filename))
                patch.set_edgecolor("black")
                patch.set_linewidth(0.5)
            for element in ["whiskers", "caps", "medians"]:
                for line in bp[element]:
                    line.set_color("black")
                    line.set_linewidth(0.8)

    ax.set_ylim(0, 100)

    group_centers = [
        group_idx * group_width + (num_models - 1) * box_width / 2
        for group_idx in range(num_groups)
    ]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(COVERAGE_LABELS, fontsize=28)
    ax.tick_params(axis="y", labelsize=28)

    if not DISABLE_LEGEND:
        legend_patches = [
            plt.Rectangle(
                (0, 0), 1, 1, facecolor=get_color(f), edgecolor="black", linewidth=0.5
            )
            for f in filenames
        ]
        ax.legend(legend_patches, model_names, loc="upper right", fontsize=22)

    plt.tight_layout()

    output_path = output_dir / "rq2_coverage.pdf"
    plt.savefig(output_path)
    plt.close()

    print(f"Saved coverage box plot to {output_path}")


def save_coverage_stats(
    stats_dir: Path = STATS_DIR,
    diagram_stats_dir: Path = DIAGRAM_STATS_DIR,
) -> None:
    """Save coverage statistics to JSON for paper writing."""
    diagram_stats_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {}
    for filename, display_name in INPUT_FILES.items():
        file_path = stats_dir / filename
        if not file_path.exists():
            continue

        data = load_results(file_path)
        distributions = data["holistic"]["distributions"]

        stats[display_name] = {}
        for cov_type, cov_label in zip(COVERAGE_TYPES, COVERAGE_LABELS):
            mean = distributions[cov_type]["mean"] * 100
            ci = (
                calculate_95_ci(
                    distributions[cov_type]["std"],
                    distributions[cov_type]["count"],
                )
                * 100
            )
            stats[display_name][cov_label] = {
                "mean_pct": round(mean, 2),
                "ci_95_pct": round(ci, 2),
            }

    output_path = diagram_stats_dir / "rq2_coverage_stats.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved coverage stats to {output_path}")


def create_coverage_chart(
    graph_type: GraphType = GRAPH_TYPE,
    stats_dir: Path = STATS_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Create coverage chart (bar graph or box plot) for all models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_data: dict[str, dict[str, dict[str, float]]] = {}
    model_names: list[str] = []
    filenames: list[str] = []

    for filename, display_name in INPUT_FILES.items():
        file_path = stats_dir / filename
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping")
            continue

        data = load_results(file_path)
        distributions = data["holistic"]["distributions"]

        model_data[filename] = {}
        for cov_type in COVERAGE_TYPES:
            model_data[filename][cov_type] = distributions[cov_type]

        model_names.append(display_name)
        filenames.append(filename)

    if graph_type == "bar_graph":
        create_coverage_bar_graph(model_data, filenames, model_names, output_dir)
    else:
        create_coverage_box_plot(model_data, filenames, model_names, output_dir)


def main() -> None:
    """Entry point."""
    create_coverage_chart(graph_type=GRAPH_TYPE)
    save_coverage_stats()


if __name__ == "__main__":
    main()
