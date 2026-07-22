"""Structured-output schema for the description-grading agent.

Deliberately free of claude-agent-sdk imports so the host (which does not
install the SDK) can import and test the validation logic.
"""

from __future__ import annotations

from typing import Any

PERCEIVED_LEVELS = ("low", "low_medium", "medium", "medium_high", "high")

STRUCTURED_OUTPUT_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "fidelity",
            "fidelity_rationale",
            "perceived_level",
            "perceived_rationale",
        ],
        "properties": {
            "fidelity": {
                "type": "integer",
                "minimum": 1,
                "maximum": 4,
                "description": (
                    "Agreement that the description accurately preserves the "
                    "tested behavior and outcome (1 = strongly disagree, "
                    "4 = strongly agree)."
                ),
            },
            "fidelity_rationale": {
                "type": "string",
                "description": (
                    "2-4 sentence justification for the fidelity grade that "
                    "cites concrete evidence from the test source that was "
                    "read (assertions, method calls, or values). Placeholders "
                    "such as 'n/a' are invalid."
                ),
            },
            "perceived_level": {
                "enum": list(PERCEIVED_LEVELS),
                "description": (
                    "The abstraction level the description reads as, judged "
                    "from the description alone."
                ),
            },
            "perceived_rationale": {
                "type": "string",
                "description": (
                    "2-4 sentence justification for the perceived abstraction "
                    "level, quoting phrasing from the description. "
                    "Placeholders such as 'n/a' are invalid."
                ),
            },
        },
    },
}


def validate_structured_grades(payload: Any) -> dict[str, Any]:
    """Re-validate the agent's structured output; belt-and-braces over the SDK.

    Returns:
        ``{"fidelity": int, "perceived_level": str}``.

    Raises:
        ValueError: When the payload does not match the schema.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"Structured output must be an object, got: {payload!r}")

    fidelity = payload.get("fidelity")
    if isinstance(fidelity, bool) or not isinstance(fidelity, int):
        raise ValueError(f"fidelity must be an integer, got: {fidelity!r}")
    if not 1 <= fidelity <= 4:
        raise ValueError(f"fidelity must be between 1 and 4, got: {fidelity}")

    perceived_level = payload.get("perceived_level")
    if perceived_level not in PERCEIVED_LEVELS:
        raise ValueError(
            f"perceived_level must be one of {PERCEIVED_LEVELS}, "
            f"got: {perceived_level!r}"
        )

    # A length floor catches placeholder rationales ("n/a", "none", "-")
    # that a plain non-empty check would let through.
    for key in ("fidelity_rationale", "perceived_rationale"):
        value = payload.get(key)
        if not isinstance(value, str) or len(value.strip()) < 20:
            raise ValueError(
                f"{key} must be a substantive rationale of at least 20 "
                f"characters, got: {value!r}"
            )

    return {"fidelity": fidelity, "perceived_level": perceived_level}
