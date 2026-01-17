"""
Comparison script for evaluation results.

Generates comparative diagrams between two sets of compiled evaluation results,
showing metrics across various dimensions like abstraction levels, focal method
buckets, and focal class counts.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
CLEANED_RESULTS_DIR = ROOT_DIR / "resources" / "cleaned_evaluation"
OUTPUT_DIR = ROOT_DIR / "resources" / "diagrams"

FILE_A = "compiled_results_out_of_box_excluded.json"
FILE_B = "compiled_results_gemini_flash_excluded.json"

# Metrics to ignore in comparisons
IGNORED_METRICS = {"localization_recall", "llm_calls"}

# Core metrics to compare (excluding ignored ones)
CORE_METRICS = [
    "obj_creation_recall",
    "obj_creation_precision",
    "assertion_recall",
    "assertion_precision",
    "callable_recall",
    "callable_precision",
    "focal_recall",
    "focal_precision",
    "class_coverage",
    "method_coverage",
    "line_coverage",
    "branch_coverage",
]

TOKEN_METRICS = ["input_tokens", "output_tokens"]


def load_results(file_path: Path) -> dict[str, Any]:
    """Load evaluation results from a JSON file."""
    with open(file_path) as f:
        return json.load(f)


def format_metric_name(metric: str) -> str:
    """Convert snake_case metric name to a human-readable label."""
    return metric.replace("_", " ").title()


def format_bucket_name(bucket: str) -> str:
    """Convert bucket key to a human-readable label."""
    replacements = {
        "one_focal": "1 Focal",
        "two_focal": "2 Focal",
        "three_to_five_focal": "3-5 Focal",
        "six_to_ten_focal": "6-10 Focal",
        "more_than_ten_focal": ">10 Focal",
        "1_focal_class": "1 Class",
        "2_focal_classes": "2 Classes",
        "3_to_5_focal_classes": "3-5 Classes",
        "high": "High",
        "medium": "Medium",
        "low": "Low",
    }
    return replacements.get(bucket, bucket.replace("_", " ").title())


def create_grouped_bar_chart(
    categories: list[str],
    values_a: list[float],
    values_b: list[float],
    label_a: str,
    label_b: str,
    title: str,
    ylabel: str,
    output_path: Path,
    figsize: tuple[int, int] = (12, 6),
    ylim: tuple[float, float] | None = None,
    show_percentage: bool = False,
) -> None:
    """Create a grouped bar chart comparing two datasets."""
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    bars_a = ax.bar(x - width / 2, values_a, width, label=label_a, color="#2E86AB")
    bars_b = ax.bar(x + width / 2, values_b, width, label=label_b, color="#A23B72")

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=10)
    ax.legend(loc="upper right", fontsize=10)

    if ylim:
        ax.set_ylim(ylim)

    # Add value labels on bars
    def add_labels(bars: Any, values: list[float]) -> None:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if show_percentage:
                label = f"{val:.1%}"
            elif val >= 1000:
                label = f"{val / 1000:.1f}k"
            else:
                label = f"{val:.2f}"
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    add_labels(bars_a, values_a)
    add_labels(bars_b, values_b)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def create_holistic_comparison(
    data_a: dict[str, Any],
    data_b: dict[str, Any],
    label_a: str,
    label_b: str,
    output_dir: Path,
) -> None:
    """Create comparison charts for holistic metrics."""
    holistic_a = data_a["holistic"]["distributions"]
    holistic_b = data_b["holistic"]["distributions"]

    # Compile rate comparison
    create_grouped_bar_chart(
        categories=["Overall"],
        values_a=[data_a["holistic"]["compile_rate"]],
        values_b=[data_b["holistic"]["compile_rate"]],
        label_a=label_a,
        label_b=label_b,
        title="Overall Compilation Rate Comparison",
        ylabel="Compilation Rate",
        output_path=output_dir / "holistic_compile_rate.png",
        figsize=(6, 5),
        ylim=(0, 1.0),
        show_percentage=True,
    )

    # Core metrics comparison (means)
    metrics = [m for m in CORE_METRICS if m in holistic_a]
    values_a = [holistic_a[m]["mean"] for m in metrics]
    values_b = [holistic_b[m]["mean"] for m in metrics]
    labels = [format_metric_name(m) for m in metrics]

    create_grouped_bar_chart(
        categories=labels,
        values_a=values_a,
        values_b=values_b,
        label_a=label_a,
        label_b=label_b,
        title="Holistic Metrics Comparison (Mean Values)",
        ylabel="Mean Value",
        output_path=output_dir / "holistic_metrics_mean.png",
        figsize=(14, 6),
        ylim=(0, 1.0),
    )

    # Token usage comparison
    token_values_a = [holistic_a[m]["mean"] for m in TOKEN_METRICS]
    token_values_b = [holistic_b[m]["mean"] for m in TOKEN_METRICS]
    token_labels = [format_metric_name(m) for m in TOKEN_METRICS]

    create_grouped_bar_chart(
        categories=token_labels,
        values_a=token_values_a,
        values_b=token_values_b,
        label_a=label_a,
        label_b=label_b,
        title="Token Usage Comparison (Mean Values)",
        ylabel="Tokens",
        output_path=output_dir / "holistic_token_usage.png",
        figsize=(8, 5),
    )

    # Average cost per sample comparison (read from distributions)
    cost_a = holistic_a.get("cost", {}).get("mean", 0.0)
    cost_b = holistic_b.get("cost", {}).get("mean", 0.0)

    # Get pricing models from summary
    pricing_a = data_a.get("summary", {}).get("pricing_model", "unknown")
    pricing_b = data_b.get("summary", {}).get("pricing_model", "unknown")

    # Create cost comparison chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1)
    width = 0.35

    bars_a = ax.bar(x - width / 2, [cost_a], width, label=label_a, color="#2E86AB")
    bars_b = ax.bar(x + width / 2, [cost_b], width, label=label_b, color="#A23B72")

    ax.set_ylabel("Cost (USD)", fontsize=11)
    ax.set_title("Average Cost per Sample", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Avg Cost"])
    ax.legend(loc="upper right", fontsize=10)

    # Add value labels
    for bar, val in zip(bars_a, [cost_a]):
        ax.annotate(
            f"${val:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    for bar, val in zip(bars_b, [cost_b]):
        ax.annotate(
            f"${val:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Add pricing model info as subtitle
    ax.text(
        0.5,
        -0.15,
        f"{label_a}: {pricing_a}\n{label_b}: {pricing_b}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="gray",
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_dir / "holistic_average_cost.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'holistic_average_cost.png'}")


def create_abstraction_level_comparison(
    data_a: dict[str, Any],
    data_b: dict[str, Any],
    label_a: str,
    label_b: str,
    output_dir: Path,
) -> None:
    """Create comparison charts for abstraction levels."""
    levels_a = data_a["abstraction_levels"]
    levels_b = data_b["abstraction_levels"]

    level_order = ["high", "medium", "low"]
    level_labels = [format_bucket_name(l) for l in level_order]

    # Compile rate comparison
    compile_a = [levels_a[l]["compile_rate"] for l in level_order]
    compile_b = [levels_b[l]["compile_rate"] for l in level_order]

    create_grouped_bar_chart(
        categories=level_labels,
        values_a=compile_a,
        values_b=compile_b,
        label_a=label_a,
        label_b=label_b,
        title="Compilation Rate by Abstraction Level",
        ylabel="Compilation Rate",
        output_path=output_dir / "abstraction_compile_rate.png",
        figsize=(8, 5),
        ylim=(0, 1.0),
        show_percentage=True,
    )

    # Key metrics per abstraction level
    key_metrics = [
        "class_coverage",
        "method_coverage",
        "line_coverage",
        "branch_coverage",
    ]
    for metric in key_metrics:
        values_a = [levels_a[l]["distributions"][metric]["mean"] for l in level_order]
        values_b = [levels_b[l]["distributions"][metric]["mean"] for l in level_order]

        create_grouped_bar_chart(
            categories=level_labels,
            values_a=values_a,
            values_b=values_b,
            label_a=label_a,
            label_b=label_b,
            title=f"{format_metric_name(metric)} by Abstraction Level",
            ylabel="Mean Value",
            output_path=output_dir / f"abstraction_{metric}.png",
            figsize=(8, 5),
            ylim=(0, 1.0),
        )


def create_focal_bucket_comparison(
    data_a: dict[str, Any],
    data_b: dict[str, Any],
    label_a: str,
    label_b: str,
    output_dir: Path,
) -> None:
    """Create comparison charts for focal method buckets."""
    buckets_a = data_a["focal_method_buckets"]
    buckets_b = data_b["focal_method_buckets"]

    bucket_order = [
        "one_focal",
        "two_focal",
        "three_to_five_focal",
        "six_to_ten_focal",
        "more_than_ten_focal",
    ]
    bucket_labels = [format_bucket_name(b) for b in bucket_order]

    # Compile rate comparison
    compile_a = [buckets_a[b]["compile_rate"] for b in bucket_order]
    compile_b = [buckets_b[b]["compile_rate"] for b in bucket_order]

    create_grouped_bar_chart(
        categories=bucket_labels,
        values_a=compile_a,
        values_b=compile_b,
        label_a=label_a,
        label_b=label_b,
        title="Compilation Rate by Focal Method Count",
        ylabel="Compilation Rate",
        output_path=output_dir / "focal_bucket_compile_rate.png",
        figsize=(10, 5),
        ylim=(0, 1.0),
        show_percentage=True,
    )

    # Key metrics per focal bucket
    key_metrics = [
        "class_coverage",
        "method_coverage",
        "line_coverage",
        "branch_coverage",
        "focal_recall",
        "focal_precision",
    ]
    for metric in key_metrics:
        values_a = [buckets_a[b]["distributions"][metric]["mean"] for b in bucket_order]
        values_b = [buckets_b[b]["distributions"][metric]["mean"] for b in bucket_order]

        create_grouped_bar_chart(
            categories=bucket_labels,
            values_a=values_a,
            values_b=values_b,
            label_a=label_a,
            label_b=label_b,
            title=f"{format_metric_name(metric)} by Focal Method Count",
            ylabel="Mean Value",
            output_path=output_dir / f"focal_bucket_{metric}.png",
            figsize=(10, 5),
            ylim=(0, 1.0),
        )


def create_focal_class_comparison(
    data_a: dict[str, Any],
    data_b: dict[str, Any],
    label_a: str,
    label_b: str,
    output_dir: Path,
) -> None:
    """Create comparison charts for focal class counts."""
    classes_a = data_a["focal_class_counts"]
    classes_b = data_b["focal_class_counts"]

    class_order = ["1_focal_class", "2_focal_classes", "3_to_5_focal_classes"]
    class_labels = [format_bucket_name(c) for c in class_order]

    # Compile rate comparison
    compile_a = [classes_a[c]["compile_rate"] for c in class_order]
    compile_b = [classes_b[c]["compile_rate"] for c in class_order]

    create_grouped_bar_chart(
        categories=class_labels,
        values_a=compile_a,
        values_b=compile_b,
        label_a=label_a,
        label_b=label_b,
        title="Compilation Rate by Focal Class Count",
        ylabel="Compilation Rate",
        output_path=output_dir / "focal_class_compile_rate.png",
        figsize=(8, 5),
        ylim=(0, 1.0),
        show_percentage=True,
    )

    # Key metrics per focal class count
    key_metrics = [
        "class_coverage",
        "method_coverage",
        "line_coverage",
        "branch_coverage",
        "focal_recall",
        "focal_precision",
    ]
    for metric in key_metrics:
        values_a = [classes_a[c]["distributions"][metric]["mean"] for c in class_order]
        values_b = [classes_b[c]["distributions"][metric]["mean"] for c in class_order]

        create_grouped_bar_chart(
            categories=class_labels,
            values_a=values_a,
            values_b=values_b,
            label_a=label_a,
            label_b=label_b,
            title=f"{format_metric_name(metric)} by Focal Class Count",
            ylabel="Mean Value",
            output_path=output_dir / f"focal_class_{metric}.png",
            figsize=(8, 5),
            ylim=(0, 1.0),
        )


def create_combined_compile_rate_chart(
    data_a: dict[str, Any],
    data_b: dict[str, Any],
    label_a: str,
    label_b: str,
    output_dir: Path,
) -> None:
    """Create a combined chart showing compile rates across all distribution types."""
    categories = []
    values_a = []
    values_b = []

    # Overall
    categories.append("Overall")
    values_a.append(data_a["holistic"]["compile_rate"])
    values_b.append(data_b["holistic"]["compile_rate"])

    # Abstraction levels
    for level in ["high", "medium", "low"]:
        categories.append(f"Abstr: {format_bucket_name(level)}")
        values_a.append(data_a["abstraction_levels"][level]["compile_rate"])
        values_b.append(data_b["abstraction_levels"][level]["compile_rate"])

    # Focal method buckets
    for bucket in ["one_focal", "two_focal", "three_to_five_focal"]:
        categories.append(f"FM: {format_bucket_name(bucket)}")
        values_a.append(data_a["focal_method_buckets"][bucket]["compile_rate"])
        values_b.append(data_b["focal_method_buckets"][bucket]["compile_rate"])

    # Focal class counts
    for cls in ["1_focal_class", "2_focal_classes"]:
        categories.append(f"FC: {format_bucket_name(cls)}")
        values_a.append(data_a["focal_class_counts"][cls]["compile_rate"])
        values_b.append(data_b["focal_class_counts"][cls]["compile_rate"])

    create_grouped_bar_chart(
        categories=categories,
        values_a=values_a,
        values_b=values_b,
        label_a=label_a,
        label_b=label_b,
        title="Compilation Rate Comparison Across All Distributions",
        ylabel="Compilation Rate",
        output_path=output_dir / "combined_compile_rates.png",
        figsize=(16, 6),
        ylim=(0, 1.0),
        show_percentage=True,
    )


def create_summary_dashboard(
    data_a: dict[str, Any],
    data_b: dict[str, Any],
    label_a: str,
    label_b: str,
    output_dir: Path,
) -> None:
    """Create a summary dashboard with key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Evaluation Results Summary Dashboard", fontsize=14, fontweight="bold")

    # Panel 1: Overall metrics
    ax1 = axes[0, 0]
    metrics = ["compile_rate", "class_coverage", "method_coverage", "line_coverage"]
    metric_labels = ["Compile Rate", "Class Cov.", "Method Cov.", "Line Cov."]
    x = np.arange(len(metrics))
    width = 0.35

    holistic_a = data_a["holistic"]
    holistic_b = data_b["holistic"]

    vals_a = [
        holistic_a["compile_rate"],
        holistic_a["distributions"]["class_coverage"]["mean"],
        holistic_a["distributions"]["method_coverage"]["mean"],
        holistic_a["distributions"]["line_coverage"]["mean"],
    ]
    vals_b = [
        holistic_b["compile_rate"],
        holistic_b["distributions"]["class_coverage"]["mean"],
        holistic_b["distributions"]["method_coverage"]["mean"],
        holistic_b["distributions"]["line_coverage"]["mean"],
    ]

    ax1.bar(x - width / 2, vals_a, width, label=label_a, color="#2E86AB")
    ax1.bar(x + width / 2, vals_b, width, label=label_b, color="#A23B72")
    ax1.set_ylabel("Value")
    ax1.set_title("Overall Metrics")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, rotation=45, ha="right")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylim(0, 1.0)

    # Panel 2: Abstraction level compile rates
    ax2 = axes[0, 1]
    levels = ["high", "medium", "low"]
    level_labels = [format_bucket_name(l) for l in levels]
    x = np.arange(len(levels))

    vals_a = [data_a["abstraction_levels"][l]["compile_rate"] for l in levels]
    vals_b = [data_b["abstraction_levels"][l]["compile_rate"] for l in levels]

    ax2.bar(x - width / 2, vals_a, width, label=label_a, color="#2E86AB")
    ax2.bar(x + width / 2, vals_b, width, label=label_b, color="#A23B72")
    ax2.set_ylabel("Compile Rate")
    ax2.set_title("Compile Rate by Abstraction Level")
    ax2.set_xticks(x)
    ax2.set_xticklabels(level_labels)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylim(0, 1.0)

    # Panel 3: Focal method bucket compile rates
    ax3 = axes[1, 0]
    buckets = ["one_focal", "two_focal", "three_to_five_focal"]
    bucket_labels = [format_bucket_name(b) for b in buckets]
    x = np.arange(len(buckets))

    vals_a = [data_a["focal_method_buckets"][b]["compile_rate"] for b in buckets]
    vals_b = [data_b["focal_method_buckets"][b]["compile_rate"] for b in buckets]

    ax3.bar(x - width / 2, vals_a, width, label=label_a, color="#2E86AB")
    ax3.bar(x + width / 2, vals_b, width, label=label_b, color="#A23B72")
    ax3.set_ylabel("Compile Rate")
    ax3.set_title("Compile Rate by Focal Method Count")
    ax3.set_xticks(x)
    ax3.set_xticklabels(bucket_labels)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.set_ylim(0, 1.0)

    # Panel 4: Coverage metrics by abstraction level (line coverage)
    ax4 = axes[1, 1]
    levels = ["high", "medium", "low"]
    level_labels = [format_bucket_name(l) for l in levels]
    x = np.arange(len(levels))

    vals_a = [
        data_a["abstraction_levels"][l]["distributions"]["line_coverage"]["mean"]
        for l in levels
    ]
    vals_b = [
        data_b["abstraction_levels"][l]["distributions"]["line_coverage"]["mean"]
        for l in levels
    ]

    ax4.bar(x - width / 2, vals_a, width, label=label_a, color="#2E86AB")
    ax4.bar(x + width / 2, vals_b, width, label=label_b, color="#A23B72")
    ax4.set_ylabel("Line Coverage")
    ax4.set_title("Line Coverage by Abstraction Level")
    ax4.set_xticks(x)
    ax4.set_xticklabels(level_labels)
    ax4.legend(loc="upper right", fontsize=8)
    ax4.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_dir / "summary_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'summary_dashboard.png'}")


LABEL_MAPPINGS = {
    "compiled_results_out_of_box.json": "Gemini CLI",
    "compiled_results_gemini_flash.json": "NL2Test (Flash)",
}


def derive_label_from_filename(filename: str) -> str:
    """Derive a human-readable label from the filename."""
    if filename in LABEL_MAPPINGS:
        return LABEL_MAPPINGS[filename]
    name = filename.replace("compiled_results_", "").replace(".json", "")
    return name.replace("_", " ").title()


def main(
    file_a: str = FILE_A,
    file_b: str = FILE_B,
    results_dir: Path = CLEANED_RESULTS_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Generate comparison diagrams between two evaluation result files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    path_a = results_dir / file_a
    path_b = results_dir / file_b

    if not path_a.exists():
        print(f"Error: File A not found: {path_a}")
        return

    if not path_b.exists():
        print(f"Error: File B not found: {path_b}")
        print("Generating diagrams for File A only...")
        data_a = load_results(path_a)
        label_a = derive_label_from_filename(file_a)
        # Could add single-file visualization here if needed
        return

    data_a = load_results(path_a)
    data_b = load_results(path_b)

    label_a = derive_label_from_filename(file_a)
    label_b = derive_label_from_filename(file_b)

    print(f"Comparing: {label_a} vs {label_b}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    create_holistic_comparison(data_a, data_b, label_a, label_b, output_dir)
    create_abstraction_level_comparison(data_a, data_b, label_a, label_b, output_dir)
    create_focal_bucket_comparison(data_a, data_b, label_a, label_b, output_dir)
    create_focal_class_comparison(data_a, data_b, label_a, label_b, output_dir)
    create_combined_compile_rate_chart(data_a, data_b, label_a, label_b, output_dir)
    create_summary_dashboard(data_a, data_b, label_a, label_b, output_dir)

    print("-" * 50)
    print(f"All diagrams saved to: {output_dir}")


if __name__ == "__main__":
    main()
