import json
import math
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from colors import DIAGRAM_STATS_DIR, INPUT_FILES, OUTPUT_DIR, STATS_DIR, get_color

ABSTRACTION_LEVELS = ["low", "medium", "high"]
ABSTRACTION_LABELS = ["Low", "Medium", "High"]

METRICS: dict[str, tuple[str, str]] = {
    "obj_creation_recall": ("Type Instantiation", "rq4_abs_type_instantiation"),
    "assertion_recall": ("Assertion Types", "rq4_abs_assertion_types"),
    "callable_recall": ("Method Calls", "rq4_abs_method_calls"),
    "focal_recall": ("Focal Methods", "rq4_abs_focal_methods"),
    "class_coverage": ("Class Coverage", "rq4_abs_class_coverage"),
    "method_coverage": ("Method Coverage", "rq4_abs_method_coverage"),
    "line_coverage": ("Line Coverage", "rq4_abs_line_coverage"),
    "branch_coverage": ("Branch Coverage", "rq4_abs_branch_coverage"),
}

GraphType = Literal["bar_graph", "line_graph"]
GRAPH_TYPE: GraphType = "bar_graph"

DISABLE_LEGEND: dict[str, bool] = {
    "obj_creation_recall": True,
    "assertion_recall": True,
    "callable_recall": True,
    "focal_recall": True,
    "class_coverage": True,
    "method_coverage": True,
    "line_coverage": True,
    "branch_coverage": True,
}

DISABLE_TITLE: dict[str, bool] = {
    "obj_creation_recall": True,
    "assertion_recall": True,
    "callable_recall": True,
    "focal_recall": True,
    "class_coverage": True,
    "method_coverage": True,
    "line_coverage": True,
    "branch_coverage": True,
}


def load_results(file_path: Path) -> dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def calculate_95_ci(std: float, count: int) -> float:
    """Calculate 95% confidence interval half-width."""
    if count == 0:
        return 0.0
    return 1.96 * (std / math.sqrt(count))


def create_bar_graph(
    metric_key: str,
    metric_label: str,
    output_name: str,
    model_data: dict[str, dict[str, dict[str, dict[str, float]]]],
    filenames: list[str],
    model_names: list[str],
    output_dir: Path,
) -> None:
    """Create grouped bar graph for a single metric across abstraction levels."""
    num_groups = len(ABSTRACTION_LABELS)
    num_models = len(filenames)
    bar_width = 0.12
    group_width = num_models * bar_width + 0.15

    fig, ax = plt.subplots(figsize=(8, 8))

    bars_list = []
    for model_idx, filename in enumerate(filenames):
        values = [
            model_data[filename][level][metric_key]["mean"] * 100
            for level in ABSTRACTION_LEVELS
        ]
        ci_errors = [
            calculate_95_ci(
                model_data[filename][level][metric_key]["std"],
                int(model_data[filename][level][metric_key]["count"]),
            )
            * 100
            for level in ABSTRACTION_LEVELS
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
    if not DISABLE_TITLE[metric_key]:
        ax.set_title(metric_label, fontsize=32)

    group_centers = [
        group_idx * group_width + (num_models - 1) * bar_width / 2
        for group_idx in range(num_groups)
    ]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(ABSTRACTION_LABELS, fontsize=34)
    ax.tick_params(axis="y", labelsize=34)

    if not DISABLE_LEGEND[metric_key]:
        ax.legend(bars_list, model_names, loc="upper right", fontsize=26)

    plt.tight_layout()

    output_path = output_dir / f"{output_name}.pdf"
    plt.savefig(output_path)
    plt.close()

    print(f"Saved {output_path}")


def create_line_graph(
    metric_key: str,
    metric_label: str,
    output_name: str,
    model_data: dict[str, dict[str, dict[str, dict[str, float]]]],
    filenames: list[str],
    model_names: list[str],
    output_dir: Path,
) -> None:
    """Create line graph for a single metric across abstraction levels."""
    fig, ax = plt.subplots(figsize=(8, 8))

    x_positions = list(range(len(ABSTRACTION_LEVELS)))

    for filename, model_name in zip(filenames, model_names):
        values = [
            model_data[filename][level][metric_key]["mean"] * 100
            for level in ABSTRACTION_LEVELS
        ]
        ci_errors = [
            calculate_95_ci(
                model_data[filename][level][metric_key]["std"],
                int(model_data[filename][level][metric_key]["count"]),
            )
            * 100
            for level in ABSTRACTION_LEVELS
        ]

        ax.errorbar(
            x_positions,
            values,
            yerr=ci_errors,
            marker="o",
            color=get_color(filename),
            linewidth=2,
            markersize=8,
            capsize=4,
            capthick=1.5,
            label=model_name,
        )

    ax.set_ylim(0, 100)
    if not DISABLE_TITLE[metric_key]:
        ax.set_title(metric_label, fontsize=32)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ABSTRACTION_LABELS, fontsize=34)
    ax.tick_params(axis="y", labelsize=34)

    if not DISABLE_LEGEND[metric_key]:
        ax.legend(loc="upper right", fontsize=26)

    plt.tight_layout()

    output_path = output_dir / f"{output_name}.pdf"
    plt.savefig(output_path)
    plt.close()

    print(f"Saved {output_path}")


def create_abstraction_sensitivity_charts(
    graph_type: GraphType = GRAPH_TYPE,
    stats_dir: Path = STATS_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Create abstraction sensitivity charts for all metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data for all models
    model_data: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    model_names: list[str] = []
    filenames: list[str] = []

    for filename, display_name in INPUT_FILES.items():
        file_path = stats_dir / filename
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping")
            continue

        data = load_results(file_path)
        abstraction_data = data.get("abstraction_levels", {})

        model_data[filename] = {}
        for level in ABSTRACTION_LEVELS:
            if level not in abstraction_data:
                print(f"Warning: {level} abstraction level not found in {filename}")
                continue
            model_data[filename][level] = abstraction_data[level]["distributions"]

        model_names.append(display_name)
        filenames.append(filename)

    # Create chart for each metric
    for metric_key, (metric_label, output_name) in METRICS.items():
        if graph_type == "bar_graph":
            create_bar_graph(
                metric_key,
                metric_label,
                output_name,
                model_data,
                filenames,
                model_names,
                output_dir,
            )
        else:
            create_line_graph(
                metric_key,
                metric_label,
                output_name,
                model_data,
                filenames,
                model_names,
                output_dir,
            )


def save_abstraction_sensitivity_stats(
    stats_dir: Path = STATS_DIR,
    diagram_stats_dir: Path = DIAGRAM_STATS_DIR,
) -> None:
    """Save abstraction sensitivity statistics to JSON for paper writing."""
    diagram_stats_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {}
    for filename, display_name in INPUT_FILES.items():
        file_path = stats_dir / filename
        if not file_path.exists():
            continue

        data = load_results(file_path)
        abstraction_data = data.get("abstraction_levels", {})

        stats[display_name] = {}
        for level, level_label in zip(ABSTRACTION_LEVELS, ABSTRACTION_LABELS):
            if level not in abstraction_data:
                continue
            distributions = abstraction_data[level]["distributions"]
            stats[display_name][level_label] = {}
            for metric_key, (metric_label, _) in METRICS.items():
                if metric_key not in distributions:
                    continue
                mean = distributions[metric_key]["mean"] * 100
                ci = (
                    calculate_95_ci(
                        distributions[metric_key]["std"],
                        int(distributions[metric_key]["count"]),
                    )
                    * 100
                )
                stats[display_name][level_label][metric_label] = {
                    "mean_pct": round(mean, 2),
                    "ci_95_pct": round(ci, 2),
                }

    output_path = diagram_stats_dir / "rq4_abstraction_sensitivity_stats.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved abstraction sensitivity stats to {output_path}")


def main() -> None:
    """Entry point."""
    create_abstraction_sensitivity_charts(graph_type=GRAPH_TYPE)
    save_abstraction_sensitivity_stats()


if __name__ == "__main__":
    main()
