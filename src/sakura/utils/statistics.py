"""Statistical utilities for computing distribution summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping


@dataclass(frozen=True)
class DistributionSummary:
    mean: float
    std: float
    min: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    max: float
    count: int
    total: float


def percentile(sorted_values: List[float], percentile_value: float) -> float:
    """Calculate percentile value from a sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (percentile_value / 100) * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return lower_value + weight * (upper_value - lower_value)


def summarize_distribution(values: List[float]) -> DistributionSummary:
    """Compute distribution summary statistics for a list of values."""
    if not values:
        return DistributionSummary(
            mean=0.0, std=0.0, min=0.0, p10=0.0, p25=0.0, p50=0.0, p75=0.0, p90=0.0, max=0.0, count=0, total=0.0
        )

    sorted_values = sorted(values)
    total_sum = sum(values)
    mean_value = total_sum / len(values)
    variance = sum((x - mean_value) ** 2 for x in values) / len(values)
    std_value = math.sqrt(variance)
    return DistributionSummary(
        mean=mean_value,
        std=std_value,
        min=sorted_values[0],
        p10=percentile(sorted_values, 10),
        p25=percentile(sorted_values, 25),
        p50=percentile(sorted_values, 50),
        p75=percentile(sorted_values, 75),
        p90=percentile(sorted_values, 90),
        max=sorted_values[-1],
        count=len(values),
        total=total_sum,
    )


def distribution_to_dict(summary: DistributionSummary) -> Dict[str, Any]:
    """Convert a DistributionSummary to a dictionary."""
    return {
        "mean": summary.mean,
        "std": summary.std,
        "min": summary.min,
        "p10": summary.p10,
        "p25": summary.p25,
        "p50": summary.p50,
        "p75": summary.p75,
        "p90": summary.p90,
        "max": summary.max,
        "count": summary.count,
        "total": summary.total,
    }


def distributions_to_dict(
    distributions: Mapping[str, DistributionSummary],
) -> Dict[str, Any]:
    """Convert a mapping of distribution summaries to a dictionary."""
    return {
        name: distribution_to_dict(summary) for name, summary in distributions.items()
    }


def build_distributions(
    metrics: Mapping[str, List[float]],
) -> Dict[str, DistributionSummary]:
    """Build distribution summaries for each metric in the mapping."""
    return {name: summarize_distribution(values) for name, values in metrics.items()}
