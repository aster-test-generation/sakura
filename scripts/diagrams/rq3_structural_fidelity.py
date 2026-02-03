import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from colors import DIAGRAM_STATS_DIR, INPUT_FILES, OUTPUT_DIR, STATS_DIR, get_color

CATEGORIES = ["obj_creation", "assertion", "callable", "focal"]
CATEGORY_LABELS = [
    "Type Instantiation",
    "Assertion Types",
    "Method Calls",
    "Focal Methods",
]
CATEGORY_OUTPUT_NAMES = [
    "type_instantiation",
    "assertion_types",
    "method_calls",
    "focal_methods",
]
METRICS = ["recall", "precision", "f1"]
METRIC_LABELS = ["Recall", "Precision", "F1"]

# If True, create 4 figures (one per category) with metrics on x-axis
# If False, create 3 figures (one per metric) with categories on x-axis
DIVIDE_BY_CATEGORY: bool = True

DISABLE_LEGEND_BY_METRIC: dict[str, bool] = {
    "recall": True,
    "precision": True,
    "f1": True,
}

DISABLE_TITLE_BY_METRIC: dict[str, bool] = {
    "recall": True,
    "precision": True,
    "f1": True,
}

DISABLE_LEGEND_BY_CATEGORY: dict[str, bool] = {
    "obj_creation": True,
    "assertion": True,
    "callable": True,
    "focal": True,
}

DISABLE_TITLE_BY_CATEGORY: dict[str, bool] = {
    "obj_creation": True,
    "assertion": True,
    "callable": True,
    "focal": True,
}


def load_results(file_path: Path) -> dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def calculate_95_ci(std: float, count: int) -> float:
    """Calculate 95% confidence interval half-width."""
    return 1.96 * (std / math.sqrt(count))


def create_graphs_by_metric(
    model_data: dict[str, dict[str, dict[str, float]]],
    filenames: list[str],
    model_names: list[str],
    output_dir: Path,
) -> None:
    """Create 3 figures (one per metric) with categories on x-axis."""
    num_categories = len(CATEGORY_LABELS)
    num_models = len(filenames)
    bar_width = 0.12
    group_width = num_models * bar_width + 0.15

    for metric, metric_label in zip(METRICS, METRIC_LABELS):
        fig, ax = plt.subplots(figsize=(8, 8))

        bars_list: list[Any] = []
        for model_idx, filename in enumerate(filenames):
            values = [
                model_data[filename][f"{cat}_{metric}"]["mean"] * 100
                for cat in CATEGORIES
            ]
            ci_errors = [
                calculate_95_ci(
                    model_data[filename][f"{cat}_{metric}"]["std"],
                    int(model_data[filename][f"{cat}_{metric}"]["count"]),
                )
                * 100
                for cat in CATEGORIES
            ]

            x_positions = [
                group_idx * group_width + model_idx * bar_width
                for group_idx in range(num_categories)
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
        if not DISABLE_TITLE_BY_METRIC[metric]:
            ax.set_title(metric_label, fontsize=32)

        group_centers = [
            group_idx * group_width + (num_models - 1) * bar_width / 2
            for group_idx in range(num_categories)
        ]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(CATEGORY_LABELS, fontsize=34, rotation=15, ha="right")
        ax.tick_params(axis="y", labelsize=34)

        if not DISABLE_LEGEND_BY_METRIC[metric]:
            ax.legend(bars_list, model_names, loc="lower right", fontsize=26)

        plt.tight_layout()

        output_path = output_dir / f"rq3_structural_{metric}.pdf"
        plt.savefig(output_path)
        plt.close()

        print(f"Saved structural fidelity {metric} graph to {output_path}")


def create_graphs_by_category(
    model_data: dict[str, dict[str, dict[str, float]]],
    filenames: list[str],
    model_names: list[str],
    output_dir: Path,
) -> None:
    """Create 4 figures (one per category) with metrics on x-axis."""
    num_metrics = len(METRIC_LABELS)
    num_models = len(filenames)
    bar_width = 0.12
    group_width = num_models * bar_width + 0.15

    for category, cat_label, cat_output in zip(
        CATEGORIES, CATEGORY_LABELS, CATEGORY_OUTPUT_NAMES
    ):
        fig, ax = plt.subplots(figsize=(8, 8))

        bars_list: list[Any] = []
        for model_idx, filename in enumerate(filenames):
            values = [
                model_data[filename][f"{category}_{metric}"]["mean"] * 100
                for metric in METRICS
            ]
            ci_errors = [
                calculate_95_ci(
                    model_data[filename][f"{category}_{metric}"]["std"],
                    int(model_data[filename][f"{category}_{metric}"]["count"]),
                )
                * 100
                for metric in METRICS
            ]

            x_positions = [
                group_idx * group_width + model_idx * bar_width
                for group_idx in range(num_metrics)
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
        if not DISABLE_TITLE_BY_CATEGORY[category]:
            ax.set_title(cat_label, fontsize=32)

        group_centers = [
            group_idx * group_width + (num_models - 1) * bar_width / 2
            for group_idx in range(num_metrics)
        ]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(METRIC_LABELS, fontsize=34)
        ax.tick_params(axis="y", labelsize=34)

        if not DISABLE_LEGEND_BY_CATEGORY[category]:
            ax.legend(bars_list, model_names, loc="upper right", fontsize=26)

        plt.tight_layout()

        output_path = output_dir / f"rq3_structural_{cat_output}.pdf"
        plt.savefig(output_path)
        plt.close()

        print(f"Saved structural fidelity {cat_label} graph to {output_path}")


def create_structural_fidelity_bar_graph(
    stats_dir: Path = STATS_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Create bar graphs of structural fidelity metrics for all models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data for all models
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
        for category in CATEGORIES:
            for metric in METRICS:
                key = f"{category}_{metric}"
                model_data[filename][key] = distributions[key]

        model_names.append(display_name)
        filenames.append(filename)

    if DIVIDE_BY_CATEGORY:
        create_graphs_by_category(model_data, filenames, model_names, output_dir)
    else:
        create_graphs_by_metric(model_data, filenames, model_names, output_dir)


def save_structural_fidelity_stats(
    stats_dir: Path = STATS_DIR,
    diagram_stats_dir: Path = DIAGRAM_STATS_DIR,
) -> None:
    """Save structural fidelity statistics to JSON for paper writing."""
    diagram_stats_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {}
    for filename, display_name in INPUT_FILES.items():
        file_path = stats_dir / filename
        if not file_path.exists():
            continue

        data = load_results(file_path)
        distributions = data["holistic"]["distributions"]

        stats[display_name] = {}
        for category, cat_label in zip(CATEGORIES, CATEGORY_LABELS):
            stats[display_name][cat_label] = {}
            for metric, metric_label in zip(METRICS, METRIC_LABELS):
                key = f"{category}_{metric}"
                mean = distributions[key]["mean"] * 100
                ci = (
                    calculate_95_ci(
                        distributions[key]["std"],
                        int(distributions[key]["count"]),
                    )
                    * 100
                )
                stats[display_name][cat_label][metric_label] = {
                    "mean_pct": round(mean, 2),
                    "ci_95_pct": round(ci, 2),
                }

    output_path = diagram_stats_dir / "rq3_structural_fidelity_stats.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved structural fidelity stats to {output_path}")


def main() -> None:
    """Entry point."""
    create_structural_fidelity_bar_graph()
    save_structural_fidelity_stats()


if __name__ == "__main__":
    main()
