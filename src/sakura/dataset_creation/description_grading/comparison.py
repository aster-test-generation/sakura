"""Comparative analysis of saved description grade files.

Human reviewers (``graded/<user>.json``) and sandboxed agents
(``agent_graded/<label>.json``) write the same grade-file schema, so this
module treats every selected name as just another rater: it joins the
selected files on the entries all of them have graded, attaches the ground
truth from the sampled CSVs, and computes standard agreement statistics for
the two rubric scales:

- fidelity: a 4-point ordinal Likert scale (1-4) with no ground truth.
- perceived abstraction: a 5-point ordinal scale (low, low/medium, medium,
  medium/high, high) coded 1-5. The true generated level occupies codes
  1/3/5 of the same axis, so one scale step equals half an abstraction
  level and a straddle pick sits exactly half a level from each neighbour.

Reported metrics: exact and adjacent (within one step) percent agreement,
mean absolute error in scale steps, Cohen's kappa (unweighted, linear- and
quadratic-weighted), and Spearman's rho for every rater pair; multi-rater
Krippendorff's alpha with the ordinal difference function; and, against
ground truth, per-rater exact/within-half-level accuracy, MAE, weighted
kappa, Spearman's rho, and a truth-by-perceived confusion matrix.
"""

from __future__ import annotations

import json
import re
import webbrowser
from itertools import combinations
from math import sqrt
from pathlib import Path
from typing import Any, Sequence

from sakura.dataset_creation.description_grading.grading_criteria import (
    FIDELITY_OPTIONS,
    PERCEIVED_OPTIONS,
)
from sakura.dataset_creation.description_grading.session import (
    Grades,
    load_grader_order,
)
from sakura.utils.file_io.structured_data_manager import StructuredDataManager
from sakura.utils.pretty.color_logger import RichLog

SCHEMA_VERSION = 3

FIDELITY_CODES: tuple[int, ...] = tuple(
    option["value"] for option in FIDELITY_OPTIONS
)
PERCEIVED_VALUES: tuple[str, ...] = tuple(
    option["value"] for option in PERCEIVED_OPTIONS
)
PERCEIVED_CODES: dict[str, int] = {
    value: index + 1 for index, value in enumerate(PERCEIVED_VALUES)
}
TRUE_LEVEL_CODES: dict[str, int] = {"low": 1, "medium": 3, "high": 5}
TRUE_LEVELS: tuple[str, ...] = ("low", "medium", "high")

_SOURCE_PREFIXES = {
    "graded": "human",
    "human": "human",
    "agent_graded": "agent",
    "agent": "agent",
}
_SOURCE_DIRS = {"human": "graded", "agent": "agent_graded"}


def resolve_grade_source(
    descriptions_dir: Path, spec: str
) -> tuple[str, str, Path]:
    """Resolve a ``--user`` spec to (name, kind, grade file path).

    A bare name is looked up in graded/ then agent_graded/; a
    ``graded:``/``human:`` or ``agent_graded:``/``agent:`` prefix pins the
    directory when the same name exists in both.
    """
    if ":" in spec:
        prefix, name = spec.split(":", 1)
        kind = _SOURCE_PREFIXES.get(prefix.strip().lower())
        if kind is None:
            raise ValueError(
                f"Unknown source prefix {prefix!r} in {spec!r}. Use one of: "
                + ", ".join(sorted(_SOURCE_PREFIXES))
            )
        path = descriptions_dir / _SOURCE_DIRS[kind] / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Grade file not found: {path}")
        return name, kind, path

    human = descriptions_dir / _SOURCE_DIRS["human"] / f"{spec}.json"
    agent = descriptions_dir / _SOURCE_DIRS["agent"] / f"{spec}.json"
    if human.is_file() and agent.is_file():
        raise ValueError(
            f"{spec!r} exists in both graded/ and agent_graded/. "
            f"Disambiguate with 'graded:{spec}' or 'agent:{spec}'."
        )
    if human.is_file():
        return spec, "human", human
    if agent.is_file():
        return spec, "agent", agent
    raise FileNotFoundError(
        f"No grade file named {spec}.json under {descriptions_dir / 'graded'} "
        f"or {descriptions_dir / 'agent_graded'}"
    )


def load_grade_file(path: Path, user: str) -> dict[int, Grades]:
    """The non-null grades of one reviewer, keyed by entry ID."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid grade file JSON: {path}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported grade file (expected schema_version {SCHEMA_VERSION}): {path}"
        )
    if data.get("user") != user:
        raise ValueError(
            f"Grade file {path} belongs to user {data.get('user')!r}, not {user!r}"
        )
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"Grade file entries must be a list: {path}")
    grades: dict[int, Grades] = {}
    for saved in entries:
        if isinstance(saved, dict) and saved.get("grades") is not None:
            grades[int(saved["id"])] = Grades.model_validate(saved["grades"])
    return grades


def load_agent_rationales(
    descriptions_dir: Path, label: str, entry_ids: Sequence[int]
) -> dict[int, dict[str, str]]:
    """Per-entry rationales from an agent run's details files, when present."""
    details_dir = descriptions_dir / _SOURCE_DIRS["agent"] / label / "details"
    if not details_dir.is_dir():
        return {}
    rationales: dict[int, dict[str, str]] = {}
    for entry_id in entry_ids:
        path = details_dir / f"entry-{entry_id:06d}.json"
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        raw = record.get("rationales")
        if isinstance(raw, dict):
            text = {
                key: value for key, value in raw.items() if isinstance(value, str)
            }
            if text:
                rationales[entry_id] = text
    return rationales


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _exact_rate(a: Sequence[int], b: Sequence[int]) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)


def _adjacent_rate(a: Sequence[int], b: Sequence[int]) -> float:
    return sum(abs(x - y) <= 1 for x, y in zip(a, b)) / len(a)


def _mae(a: Sequence[int], b: Sequence[int]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def _spearman_rho(a: Sequence[int], b: Sequence[int]) -> float | None:
    """Spearman's rho as Pearson correlation on tie-averaged ranks."""

    def ranks(values: Sequence[int]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        start = 0
        while start < len(order):
            stop = start
            while (
                stop + 1 < len(order)
                and values[order[stop + 1]] == values[order[start]]
            ):
                stop += 1
            average = (start + stop) / 2 + 1
            for position in range(start, stop + 1):
                result[order[position]] = average
            start = stop + 1
        return result

    rank_a, rank_b = ranks(a), ranks(b)
    mean_a = sum(rank_a) / len(rank_a)
    mean_b = sum(rank_b) / len(rank_b)
    covariance = sum(
        (x - mean_a) * (y - mean_b) for x, y in zip(rank_a, rank_b)
    )
    var_a = sum((x - mean_a) ** 2 for x in rank_a)
    var_b = sum((y - mean_b) ** 2 for y in rank_b)
    if var_a == 0 or var_b == 0:
        return None
    return covariance / sqrt(var_a * var_b)


def _cohen_kappa(
    a: Sequence[int],
    b: Sequence[int],
    categories: Sequence[int],
    weighting: str | None = None,
) -> float | None:
    """Cohen's kappa; ``weighting`` is None, "linear", or "quadratic"."""
    index = {category: i for i, category in enumerate(categories)}
    size = len(categories)
    total = len(a)
    observed = [[0.0] * size for _ in range(size)]
    for x, y in zip(a, b):
        observed[index[x]][index[y]] += 1.0 / total
    row_totals = [sum(row) for row in observed]
    col_totals = [sum(observed[i][j] for i in range(size)) for j in range(size)]

    def weight(i: int, j: int) -> float:
        if weighting == "linear":
            return abs(i - j) / (size - 1)
        if weighting == "quadratic":
            return ((i - j) / (size - 1)) ** 2
        return 0.0 if i == j else 1.0

    disagreement = sum(
        weight(i, j) * observed[i][j] for i in range(size) for j in range(size)
    )
    expected = sum(
        weight(i, j) * row_totals[i] * col_totals[j]
        for i in range(size)
        for j in range(size)
    )
    if expected == 0:
        return None
    return 1.0 - disagreement / expected


def _krippendorff_alpha_ordinal(
    values_by_unit: Sequence[Sequence[int]], categories: Sequence[int]
) -> float | None:
    """Krippendorff's alpha with the ordinal difference function.

    Each inner sequence holds the codes all raters assigned to one unit;
    units with fewer than two ratings are ignored per the standard method.
    """
    index = {category: i for i, category in enumerate(categories)}
    size = len(categories)
    coincidence = [[0.0] * size for _ in range(size)]
    for values in values_by_unit:
        count = len(values)
        if count < 2:
            continue
        for i in range(count):
            for j in range(count):
                if i != j:
                    coincidence[index[values[i]]][index[values[j]]] += 1.0 / (
                        count - 1
                    )
    totals = [sum(row) for row in coincidence]
    grand_total = sum(totals)
    if grand_total <= 1:
        return None

    def delta_squared(low: int, high: int) -> float:
        between = sum(totals[g] for g in range(low, high + 1))
        return (between - (totals[low] + totals[high]) / 2.0) ** 2

    observed = sum(
        coincidence[c][d] * delta_squared(c, d)
        for c in range(size)
        for d in range(c + 1, size)
    )
    expected = sum(
        totals[c] * totals[d] * delta_squared(c, d)
        for c in range(size)
        for d in range(c + 1, size)
    )
    if expected == 0:
        return None
    return 1.0 - (grand_total - 1.0) * observed / expected


def _pairwise_metrics(
    codes_by_user: dict[str, dict[int, int]],
    common_ids: Sequence[int],
    categories: Sequence[int],
) -> list[dict[str, Any]]:
    rows = []
    for user_a, user_b in combinations(codes_by_user, 2):
        a = [codes_by_user[user_a][entry_id] for entry_id in common_ids]
        b = [codes_by_user[user_b][entry_id] for entry_id in common_ids]
        rows.append(
            {
                "a": user_a,
                "b": user_b,
                "n": len(common_ids),
                "exact": _round(_exact_rate(a, b)),
                "adjacent": _round(_adjacent_rate(a, b)),
                "mae": _round(_mae(a, b)),
                "kappa": _round(_cohen_kappa(a, b, categories)),
                "kappa_linear": _round(_cohen_kappa(a, b, categories, "linear")),
                "kappa_quadratic": _round(
                    _cohen_kappa(a, b, categories, "quadratic")
                ),
                "spearman": _round(_spearman_rho(a, b)),
            }
        )
    return rows


def _truth_metrics(
    perceived: Sequence[str], truth: Sequence[str]
) -> dict[str, Any]:
    codes = [PERCEIVED_CODES[value] for value in perceived]
    truth_codes = [TRUE_LEVEL_CODES[value] for value in truth]
    categories = sorted(PERCEIVED_CODES.values())
    confusion = [[0] * len(PERCEIVED_VALUES) for _ in TRUE_LEVELS]
    truth_row = {level: i for i, level in enumerate(TRUE_LEVELS)}
    for perceived_value, truth_value in zip(perceived, truth):
        confusion[truth_row[truth_value]][
            PERCEIVED_CODES[perceived_value] - 1
        ] += 1
    return {
        "n": len(codes),
        "exact": _round(_exact_rate(codes, truth_codes)),
        "within_half_level": _round(_adjacent_rate(codes, truth_codes)),
        "mae": _round(_mae(codes, truth_codes)),
        "spearman": _round(_spearman_rho(codes, truth_codes)),
        "kappa_linear": _round(
            _cohen_kappa(codes, truth_codes, categories, "linear")
        ),
        "kappa_quadratic": _round(
            _cohen_kappa(codes, truth_codes, categories, "quadratic")
        ),
        "confusion": confusion,
    }


def build_comparison_payload(
    user_specs: Sequence[str], descriptions_dir: Path, repo_root: Path
) -> dict[str, Any]:
    """The JSON-serializable document the comparison report renders."""
    if not user_specs:
        raise ValueError("At least one --user is required.")

    users: list[str] = []
    user_meta: dict[str, dict[str, Any]] = {}
    grades_by_user: dict[str, dict[int, Grades]] = {}
    for spec in user_specs:
        name, kind, path = resolve_grade_source(descriptions_dir, spec)
        if name in grades_by_user:
            raise ValueError(f"Duplicate user in selection: {name}")
        grades = load_grade_file(path, name)
        users.append(name)
        grades_by_user[name] = grades
        try:
            display_path = path.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            display_path = str(path.resolve())
        user_meta[name] = {
            "kind": kind,
            "path": display_path,
            "graded_count": len(grades),
        }

    ordered_entries = load_grader_order(descriptions_dir)
    common_ids = [
        entry.id
        for entry in ordered_entries
        if all(entry.id in grades_by_user[user] for user in users)
    ]
    if not common_ids:
        counts = ", ".join(
            f"{user}={user_meta[user]['graded_count']}" for user in users
        )
        raise ValueError(
            f"No entries are graded by every selected user (graded counts: {counts})."
        )

    rationales_by_user = {
        user: load_agent_rationales(descriptions_dir, user, common_ids)
        for user in users
        if user_meta[user]["kind"] == "agent"
    }

    # Imported here because the context service pulls in cldk, which is far
    # too heavy for module import time.
    from sakura.dataset_creation.description_grading.context import (
        DescriptionContextService,
    )

    context_service = DescriptionContextService(
        repo_root / "resources" / "datasets",
        repo_root / "resources" / "analysis",
    )

    entries_by_id = {entry.id: entry for entry in ordered_entries}
    positions = {entry.id: index for index, entry in enumerate(ordered_entries)}
    entry_payloads = []
    for entry_id in common_ids:
        entry = entries_by_id[entry_id]
        try:
            code_context = context_service.render_entry(entry)
        except Exception as exc:  # noqa: BLE001 - the report degrades gracefully
            code_context = None
            RichLog.warn(f"Could not render code context for entry {entry_id}: {exc}")
        payload: dict[str, Any] = {
            "id": entry.id,
            "position": positions[entry.id],
            "project_name": entry.project_name,
            "qualified_class_name": entry.qualified_class_name,
            "method_signature": entry.method_signature,
            "description": entry.description,
            "is_bdd": entry.is_bdd,
            "true_level": entry.abstraction_level.value,
            "code_context": code_context,
            "grades": {
                user: grades_by_user[user][entry_id].model_dump()
                for user in users
            },
        }
        entry_rationales = {
            user: per_user[entry_id]
            for user, per_user in rationales_by_user.items()
            if entry_id in per_user
        }
        if entry_rationales:
            payload["rationales"] = entry_rationales
        entry_payloads.append(payload)

    fidelity_codes = {
        user: {
            entry_id: grades_by_user[user][entry_id].fidelity
            for entry_id in common_ids
        }
        for user in users
    }
    perceived_codes = {
        user: {
            entry_id: PERCEIVED_CODES[
                grades_by_user[user][entry_id].perceived_level
            ]
            for entry_id in common_ids
        }
        for user in users
    }
    truth_values = [
        entries_by_id[entry_id].abstraction_level.value for entry_id in common_ids
    ]

    fidelity_categories = list(FIDELITY_CODES)
    perceived_categories = sorted(PERCEIVED_CODES.values())
    summary = {
        "n_common": len(common_ids),
        "fidelity": {
            "per_user": {
                user: {
                    "mean": _round(
                        sum(fidelity_codes[user].values()) / len(common_ids)
                    ),
                    "counts": {
                        str(value): sum(
                            code == value
                            for code in fidelity_codes[user].values()
                        )
                        for value in FIDELITY_CODES
                    },
                }
                for user in users
            },
            "pairwise": _pairwise_metrics(
                fidelity_codes, common_ids, fidelity_categories
            ),
            "krippendorff_alpha": _round(
                _krippendorff_alpha_ordinal(
                    [
                        [fidelity_codes[user][entry_id] for user in users]
                        for entry_id in common_ids
                    ],
                    fidelity_categories,
                )
            ),
        },
        "perceived": {
            "per_user": {
                user: {
                    "counts": {
                        value: sum(
                            grades_by_user[user][entry_id].perceived_level
                            == value
                            for entry_id in common_ids
                        )
                        for value in PERCEIVED_VALUES
                    }
                }
                for user in users
            },
            "pairwise": _pairwise_metrics(
                perceived_codes, common_ids, perceived_categories
            ),
            "krippendorff_alpha": _round(
                _krippendorff_alpha_ordinal(
                    [
                        [perceived_codes[user][entry_id] for user in users]
                        for entry_id in common_ids
                    ],
                    perceived_categories,
                )
            ),
            "vs_truth": {
                user: _truth_metrics(
                    [
                        grades_by_user[user][entry_id].perceived_level
                        for entry_id in common_ids
                    ],
                    truth_values,
                )
                for user in users
            },
        },
    }

    return {
        "users": users,
        "user_meta": user_meta,
        "entries": entry_payloads,
        "summary": summary,
    }


def run_grade_comparison(
    *,
    users: Sequence[str],
    repo_root: Path,
    output: Path | None = None,
    open_browser: bool = True,
) -> dict[str, Any]:
    """Build the comparison report HTML and return its path and entry count."""
    from sakura.dataset_creation.description_grading.comparison_ui import (
        render_report,
    )

    descriptions_dir = repo_root / "outputs" / "descriptions_sample"
    payload = build_comparison_payload(users, descriptions_dir, repo_root)
    html = render_report(payload)

    if output is None:
        slug = "-vs-".join(
            re.sub(r"[^A-Za-z0-9_.-]+", "-", user) for user in payload["users"]
        )
        output = descriptions_dir / "comparison" / f"{slug}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    StructuredDataManager._atomic_write_text(output, html)
    if open_browser:
        webbrowser.open(output.resolve().as_uri())
    return {
        "path": output,
        "users": payload["users"],
        "n_common": payload["summary"]["n_common"],
    }
