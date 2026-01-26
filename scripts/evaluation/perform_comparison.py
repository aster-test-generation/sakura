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
CLEANED_RESULTS_DIR = ROOT_DIR / "outputs" / "evaluation_stats"
OUTPUT_DIR = ROOT_DIR / "outputs" / "diagrams"

INPUT_FILES: dict[str, str] = {
    "gemini_cli_flash_eval.json": "Gemini CLI (Flash 2.5)",
    "gemini_cli_pro_eval.json": "Gemini CLI (Pro 2.5)",
    "nl2test_gemini_flash_eval.json": "NL2Test (Flash 2.5)",
    "nl2test_gemini_pro_eval.json": "NL2Test (Pro 2.5)",
    "nl2test_qwen3_eval.json": "NL2Test (Qwen3 Coder)",
    "nl2test_devstral_eval.json": "NL2Test (Devstral 2 Small)",
    # "gemini_cli_200_eval.json": "Gemini CLI (Pro 2.5)",
    # "nl2test_subset_40_gemini_flash_eval.json": "NL2Test (Gemini Flash 2.5)",
    # "nl2test_subset_40_gemini_pro_low_eval.json": "NL2Test (Gemini Pro 2.5)",
    # "nl2test_subset_40_deepseek_v3.2_eval.json": "NL2Test (DeepSeek V3.2)",
    # "nl2test_subset_40_devstral_eval.json": "NL2Test (Devstral)",
    # "nl2test_subset_40_minimax_eval.json": "NL2Test (Minimax M2.1)",
    # "nl2test_subset_40_mimo_v2_flash_eval.json": "NL2Test (Mimo V2 Flash)",
    # "nl2test_subset_40_qwen3_coder_eval.json": "NL2Test (Qwen3-Coder)",
}

# Color palette for multiple datasets
COLORS = [
    "#2E86AB",
    "#A23B72",
    "#F18F01",
    "#C73E1D",
    "#3B1F2B",
    "#95C623",
    "#5C4D7D",
    "#2A9D8F",
]

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
    all_values: list[list[float]],
    labels: list[str],
    title: str,
    ylabel: str,
    output_path: Path,
    figsize: tuple[int, int] = (12, 6),
    ylim: tuple[float, float] | None = None,
    show_percentage: bool = False,
) -> None:
    """Create a grouped bar chart comparing multiple datasets."""
    n_datasets = len(all_values)
    x = np.arange(len(categories))
    width = 0.8 / n_datasets

    fig, ax = plt.subplots(figsize=figsize)

    all_bars = []
    for i, (values, label) in enumerate(zip(all_values, labels)):
        offset = (i - (n_datasets - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width, label=label, color=COLORS[i % len(COLORS)]
        )
        all_bars.append(bars)

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

    for bars, values in zip(all_bars, all_values):
        add_labels(bars, values)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def create_holistic_comparison(
    all_data: list[dict[str, Any]],
    labels: list[str],
    output_dir: Path,
) -> None:
    """Create comparison charts for holistic metrics."""
    all_holistic = [d["holistic"]["distributions"] for d in all_data]

    # Compile rate comparison
    create_grouped_bar_chart(
        categories=["Overall"],
        all_values=[[d["holistic"]["compile_rate"]] for d in all_data],
        labels=labels,
        title="Overall Compilation Rate Comparison",
        ylabel="Compilation Rate",
        output_path=output_dir / "holistic_compile_rate.png",
        figsize=(6, 5),
        ylim=(0, 1.0),
        show_percentage=True,
    )

    # Core metrics comparison (means)
    metrics = [m for m in CORE_METRICS if m in all_holistic[0]]
    metric_labels = [format_metric_name(m) for m in metrics]
    all_values = [[h[m]["mean"] for m in metrics] for h in all_holistic]

    create_grouped_bar_chart(
        categories=metric_labels,
        all_values=all_values,
        labels=labels,
        title="Holistic Metrics Comparison (Mean Values)",
        ylabel="Mean Value",
        output_path=output_dir / "holistic_metrics_mean.png",
        figsize=(14, 6),
        ylim=(0, 1.0),
    )

    # Token usage comparison
    token_labels = [format_metric_name(m) for m in TOKEN_METRICS]
    all_token_values = [[h[m]["mean"] for m in TOKEN_METRICS] for h in all_holistic]

    create_grouped_bar_chart(
        categories=token_labels,
        all_values=all_token_values,
        labels=labels,
        title="Token Usage Comparison (Mean Values)",
        ylabel="Tokens",
        output_path=output_dir / "holistic_token_usage.png",
        figsize=(8, 5),
    )

    # Average cost per sample comparison
    all_costs = [h.get("cost", {}).get("mean", 0.0) for h in all_holistic]
    all_pricing = [
        d.get("summary", {}).get("pricing_model", "unknown") for d in all_data
    ]

    # Create cost comparison chart
    n_datasets = len(all_data)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1)
    width = 0.8 / n_datasets

    all_bars = []
    for i, (cost, label) in enumerate(zip(all_costs, labels)):
        offset = (i - (n_datasets - 1) / 2) * width
        bars = ax.bar(
            x + offset, [cost], width, label=label, color=COLORS[i % len(COLORS)]
        )
        all_bars.append(bars)

    ax.set_ylabel("Cost (USD)", fontsize=11)
    ax.set_title("Average Cost per Sample", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Avg Cost"])
    ax.legend(loc="upper right", fontsize=10)

    # Add value labels
    for bars, cost in zip(all_bars, all_costs):
        for bar in bars:
            ax.annotate(
                f"${cost:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # Add pricing model info as subtitle
    pricing_info = "\n".join(f"{lbl}: {p}" for lbl, p in zip(labels, all_pricing))
    ax.text(
        0.5,
        -0.15,
        pricing_info,
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
    all_data: list[dict[str, Any]],
    labels: list[str],
    output_dir: Path,
) -> None:
    """Create comparison charts for abstraction levels."""
    all_levels = [d["abstraction_levels"] for d in all_data]

    level_order = ["high", "medium", "low"]
    level_labels = [format_bucket_name(lvl) for lvl in level_order]

    # Compile rate comparison
    all_compile = [
        [lvls[lvl]["compile_rate"] for lvl in level_order] for lvls in all_levels
    ]

    create_grouped_bar_chart(
        categories=level_labels,
        all_values=all_compile,
        labels=labels,
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
        all_values = [
            [lvls[lvl]["distributions"][metric]["mean"] for lvl in level_order]
            for lvls in all_levels
        ]

        create_grouped_bar_chart(
            categories=level_labels,
            all_values=all_values,
            labels=labels,
            title=f"{format_metric_name(metric)} by Abstraction Level",
            ylabel="Mean Value",
            output_path=output_dir / f"abstraction_{metric}.png",
            figsize=(8, 5),
            ylim=(0, 1.0),
        )


def create_focal_bucket_comparison(
    all_data: list[dict[str, Any]],
    labels: list[str],
    output_dir: Path,
) -> None:
    """Create comparison charts for focal method buckets."""
    all_buckets = [d["focal_method_buckets"] for d in all_data]

    # Only include buckets present in ALL datasets
    preferred_order = [
        "one_focal",
        "two_focal",
        "three_to_five_focal",
        "six_to_ten_focal",
        "more_than_ten_focal",
    ]
    common_buckets = set.intersection(*[set(bkts.keys()) for bkts in all_buckets])
    bucket_order = [b for b in preferred_order if b in common_buckets]
    if not bucket_order:
        print("Warning: No common focal method buckets found, skipping comparison.")
        return
    bucket_labels = [format_bucket_name(b) for b in bucket_order]

    # Compile rate comparison
    all_compile = [
        [bkts[b]["compile_rate"] for b in bucket_order] for bkts in all_buckets
    ]

    create_grouped_bar_chart(
        categories=bucket_labels,
        all_values=all_compile,
        labels=labels,
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
        all_values = [
            [bkts[b]["distributions"][metric]["mean"] for b in bucket_order]
            for bkts in all_buckets
        ]

        create_grouped_bar_chart(
            categories=bucket_labels,
            all_values=all_values,
            labels=labels,
            title=f"{format_metric_name(metric)} by Focal Method Count",
            ylabel="Mean Value",
            output_path=output_dir / f"focal_bucket_{metric}.png",
            figsize=(10, 5),
            ylim=(0, 1.0),
        )


def create_focal_class_comparison(
    all_data: list[dict[str, Any]],
    labels: list[str],
    output_dir: Path,
) -> None:
    """Create comparison charts for focal class counts."""
    all_classes = [d["focal_class_counts"] for d in all_data]

    # Only include class counts present in ALL datasets
    preferred_order = ["1_focal_class", "2_focal_classes", "3_to_5_focal_classes"]
    common_classes = set.intersection(*[set(cls.keys()) for cls in all_classes])
    class_order = [c for c in preferred_order if c in common_classes]
    if not class_order:
        print("Warning: No common focal class counts found, skipping comparison.")
        return
    class_labels = [format_bucket_name(c) for c in class_order]

    # Compile rate comparison
    all_compile = [[cls[c]["compile_rate"] for c in class_order] for cls in all_classes]

    create_grouped_bar_chart(
        categories=class_labels,
        all_values=all_compile,
        labels=labels,
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
        all_values = [
            [cls[c]["distributions"][metric]["mean"] for c in class_order]
            for cls in all_classes
        ]

        create_grouped_bar_chart(
            categories=class_labels,
            all_values=all_values,
            labels=labels,
            title=f"{format_metric_name(metric)} by Focal Class Count",
            ylabel="Mean Value",
            output_path=output_dir / f"focal_class_{metric}.png",
            figsize=(8, 5),
            ylim=(0, 1.0),
        )


def create_combined_compile_rate_chart(
    all_data: list[dict[str, Any]],
    labels: list[str],
    output_dir: Path,
) -> None:
    """Create a combined chart showing compile rates across all distribution types."""
    # Determine common keys across all datasets
    all_levels = [d["abstraction_levels"] for d in all_data]
    all_buckets = [d["focal_method_buckets"] for d in all_data]
    all_classes = [d["focal_class_counts"] for d in all_data]

    common_levels = set.intersection(*[set(lvl.keys()) for lvl in all_levels])
    common_buckets = set.intersection(*[set(bkt.keys()) for bkt in all_buckets])
    common_classes = set.intersection(*[set(cls.keys()) for cls in all_classes])

    level_order = [lvl for lvl in ["high", "medium", "low"] if lvl in common_levels]
    bucket_order = [
        b
        for b in ["one_focal", "two_focal", "three_to_five_focal"]
        if b in common_buckets
    ]
    class_order = [
        c for c in ["1_focal_class", "2_focal_classes"] if c in common_classes
    ]

    categories = ["Overall"]
    for level in level_order:
        categories.append(f"Abstr: {format_bucket_name(level)}")
    for bucket in bucket_order:
        categories.append(f"FM: {format_bucket_name(bucket)}")
    for cls in class_order:
        categories.append(f"FC: {format_bucket_name(cls)}")

    # Build values for each dataset
    all_values = []
    for data in all_data:
        values = [data["holistic"]["compile_rate"]]
        for level in level_order:
            values.append(data["abstraction_levels"][level]["compile_rate"])
        for bucket in bucket_order:
            values.append(data["focal_method_buckets"][bucket]["compile_rate"])
        for cls in class_order:
            values.append(data["focal_class_counts"][cls]["compile_rate"])
        all_values.append(values)

    create_grouped_bar_chart(
        categories=categories,
        all_values=all_values,
        labels=labels,
        title="Compilation Rate Comparison Across All Distributions",
        ylabel="Compilation Rate",
        output_path=output_dir / "combined_compile_rates.png",
        figsize=(16, 6),
        ylim=(0, 1.0),
        show_percentage=True,
    )


def create_summary_dashboard(
    all_data: list[dict[str, Any]],
    labels: list[str],
    output_dir: Path,
) -> None:
    """Create a summary dashboard with key metrics."""
    n_datasets = len(all_data)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Evaluation Results Summary Dashboard", fontsize=14, fontweight="bold")

    width = 0.8 / n_datasets

    # Determine common keys across all datasets
    all_levels_data = [d["abstraction_levels"] for d in all_data]
    all_buckets_data = [d["focal_method_buckets"] for d in all_data]

    common_levels = set.intersection(*[set(lvl.keys()) for lvl in all_levels_data])
    common_buckets = set.intersection(*[set(bkt.keys()) for bkt in all_buckets_data])

    levels = [lvl for lvl in ["high", "medium", "low"] if lvl in common_levels]
    buckets = [
        b
        for b in ["one_focal", "two_focal", "three_to_five_focal"]
        if b in common_buckets
    ]

    # Panel 1: Overall metrics
    ax1 = axes[0, 0]
    metrics = ["compile_rate", "class_coverage", "method_coverage", "line_coverage"]
    metric_labels = ["Compile Rate", "Class Cov.", "Method Cov.", "Line Cov."]
    x = np.arange(len(metrics))

    for i, (data, label) in enumerate(zip(all_data, labels)):
        holistic = data["holistic"]
        vals = [
            holistic["compile_rate"],
            holistic["distributions"]["class_coverage"]["mean"],
            holistic["distributions"]["method_coverage"]["mean"],
            holistic["distributions"]["line_coverage"]["mean"],
        ]
        offset = (i - (n_datasets - 1) / 2) * width
        ax1.bar(x + offset, vals, width, label=label, color=COLORS[i % len(COLORS)])

    ax1.set_ylabel("Value")
    ax1.set_title("Overall Metrics")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, rotation=45, ha="right")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylim(0, 1.0)

    # Panel 2: Abstraction level compile rates
    ax2 = axes[0, 1]
    level_labels = [format_bucket_name(lvl) for lvl in levels]
    x = np.arange(len(levels))

    for i, (data, label) in enumerate(zip(all_data, labels)):
        vals = [data["abstraction_levels"][lvl]["compile_rate"] for lvl in levels]
        offset = (i - (n_datasets - 1) / 2) * width
        ax2.bar(x + offset, vals, width, label=label, color=COLORS[i % len(COLORS)])

    ax2.set_ylabel("Compile Rate")
    ax2.set_title("Compile Rate by Abstraction Level")
    ax2.set_xticks(x)
    ax2.set_xticklabels(level_labels)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylim(0, 1.0)

    # Panel 3: Focal method bucket compile rates
    ax3 = axes[1, 0]
    bucket_labels = [format_bucket_name(b) for b in buckets]
    x = np.arange(len(buckets))

    for i, (data, label) in enumerate(zip(all_data, labels)):
        vals = [data["focal_method_buckets"][b]["compile_rate"] for b in buckets]
        offset = (i - (n_datasets - 1) / 2) * width
        ax3.bar(x + offset, vals, width, label=label, color=COLORS[i % len(COLORS)])

    ax3.set_ylabel("Compile Rate")
    ax3.set_title("Compile Rate by Focal Method Count")
    ax3.set_xticks(x)
    ax3.set_xticklabels(bucket_labels)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.set_ylim(0, 1.0)

    # Panel 4: Coverage metrics by abstraction level (line coverage)
    ax4 = axes[1, 1]
    level_labels = [format_bucket_name(lvl) for lvl in levels]
    x = np.arange(len(levels))

    for i, (data, label) in enumerate(zip(all_data, labels)):
        vals = [
            data["abstraction_levels"][lvl]["distributions"]["line_coverage"]["mean"]
            for lvl in levels
        ]
        offset = (i - (n_datasets - 1) / 2) * width
        ax4.bar(x + offset, vals, width, label=label, color=COLORS[i % len(COLORS)])

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


def main(
    input_files: dict[str, str] = INPUT_FILES,
    results_dir: Path = CLEANED_RESULTS_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Generate comparison diagrams for multiple evaluation result files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate and load all input files
    all_data = []
    labels = []
    for filename, label in input_files.items():
        path = results_dir / filename
        if not path.exists():
            print(f"Error: File not found: {path}")
            continue
        all_data.append(load_results(path))
        labels.append(label)

    if len(all_data) < 2:
        print(
            "Error: Need at least 2 valid input files to generate comparison diagrams."
        )
        return

    print(f"Comparing: {', '.join(labels)}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    create_holistic_comparison(all_data, labels, output_dir)
    create_abstraction_level_comparison(all_data, labels, output_dir)
    create_focal_bucket_comparison(all_data, labels, output_dir)
    create_focal_class_comparison(all_data, labels, output_dir)
    create_combined_compile_rate_chart(all_data, labels, output_dir)
    create_summary_dashboard(all_data, labels, output_dir)

    print("-" * 50)
    print(f"All diagrams saved to: {output_dir}")


if __name__ == "__main__":
    main()
