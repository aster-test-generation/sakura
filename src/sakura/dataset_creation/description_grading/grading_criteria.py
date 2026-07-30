"""Single source of truth for the description-grading rubric text.

Every reviewer must be shown the identical rubric, so each label, anchor,
and level definition used by the web grader UI (web_ui.py) lives here and
only here.
"""

from __future__ import annotations

from typing import Any

FIDELITY_HELP = (
    "How strongly do you agree that the description accurately preserves the "
    "tested behavior and outcome, judged at the abstraction level the "
    "description reads as? Detail the level legitimately abstracts away is "
    "not an omission."
)

FIDELITY_OPTIONS: list[dict[str, Any]] = [
    {
        "value": 1,
        "label": "Strongly Disagree",
        "anchor": "The description contradicts or entirely misses the tested behavior.",
    },
    {
        "value": 2,
        "label": "Disagree",
        "anchor": (
            "Inaccuracies, or omissions its abstraction level does not "
            "justify, would mislead a reader about what the test verifies."
        ),
    },
    {
        "value": 3,
        "label": "Agree",
        "anchor": (
            "The description conveys what the test verifies, with only minor "
            "inaccuracies or omissions beyond what its abstraction level "
            "justifies."
        ),
    },
    {
        "value": 4,
        "label": "Strongly Agree",
        "anchor": (
            "At its abstraction level, the description fully preserves the "
            "essential behavior and outcome, including order where it "
            "matters."
        ),
    },
]

PERCEIVED_HELP = (
    "Judged from the description alone, which abstraction level does it read "
    "as? The dashed options mark a description that straddles two levels "
    "without fitting either."
)

PERCEIVED_OPTIONS: list[dict[str, Any]] = [
    {
        "value": "low",
        "label": "Low",
        "anchor": "Reads clearly as a low-level, implementation-complete description.",
    },
    {
        "value": "low_medium",
        "label": "Low / Medium",
        "straddle": True,
        "anchor": (
            "Straddles low and medium. The description mixes exact "
            "implementation detail with architectural phrasing and does not "
            "settle at either level."
        ),
    },
    {
        "value": "medium",
        "label": "Medium",
        "anchor": "Reads clearly as a medium-level, architectural description.",
    },
    {
        "value": "medium_high",
        "label": "Medium / High",
        "straddle": True,
        "anchor": (
            "Straddles medium and high. The description mixes architectural "
            "phrasing with business-level intent and does not settle at "
            "either level."
        ),
    },
    {
        "value": "high",
        "label": "High",
        "anchor": "Reads clearly as a high-level, business-focused description.",
    },
]

PERCEIVED_VALUES: tuple[str, ...] = tuple(
    option["value"] for option in PERCEIVED_OPTIONS
)

LEVEL_DEFINITIONS: dict[str, str] = {
    "low": (
        "The description reads as an implementation-complete specification: a "
        "developer could reconstruct the test almost line by line from it. It "
        "names the exact classes, methods, and variables from the code and "
        "gives the exact literal value of every input. Helper methods are "
        "fully unwrapped, with their logic inlined step by step. Calls into "
        "application code are stated as exact method invocations with their "
        "arguments, but what happens inside those methods is not described. "
        "Every call in a method chain is enumerated in order, and every "
        "assertion is listed with its exact API and expected value."
    ),
    "medium": (
        "The description reads as architectural guidance: it says what the "
        "test does and why, but leaves the exact code to the implementer. The "
        "machinery of the test stays visible, meaning components are invoked "
        'and their observable effects are stated, and technical vocabulary '
        'such as "repository", "mock", or "null" may appear. Classes and '
        'methods are referred to by semantic descriptors ("the payment '
        'service") rather than exact names; exact variable names appear only '
        "where needed to disambiguate between similar entities. Helper "
        "methods are described by their intent, without implementation "
        "details. Application code is never looked into; the description says "
        "only that it is invoked and what effects are observable from the "
        "outside. Method chains are collapsed into the logical operation they "
        "perform, inputs are characterized by their type, constraints, or "
        "general characteristics rather than literal values, and each "
        "assertion is described by what it intends to verify, with references "
        "to the relevant variables."
    ),
    "high": (
        "The description reads as a business-level requirement: it states "
        "what the system should do in domain terms, as if no code existed. "
        "The machinery of the test is fully hidden, and technical vocabulary "
        'never appears: words like "class", "method", "variable", '
        '"repository", "database", or "null" are replaced by business '
        'equivalents such as "system records", "a unique reference", or '
        '"missing". It speaks only of business entities and concepts; the '
        "only technical content is the closing framework and library listing. "
        "Helper methods and individual method calls are invisible, so the "
        "description mentions only the states or preconditions they produce, "
        "and a chain of calls is reduced to its final state. Inputs are "
        'described as domain archetypes and scenarios ("a customer with an '
        'overdue invoice"), and verification is framed as observable business '
        'outcomes ("the transaction is declined") rather than checks on '
        "variables or values."
    ),
}

INVARIANT_NOTE = (
    "Invariant across levels: each description is one self-contained "
    "paragraph, leaves the test name unspecified, and ends by listing the "
    "testing framework, assertion library, and mocking library. Any level may "
    "also use an exact literal value when looking up pre-seeded data that "
    'exists outside the test, such as "the owner with ID 6". These traits '
    "carry no level signal."
)


def criteria_payload() -> dict[str, Any]:
    """The JSON-serializable rubric bundle embedded into agent task specs."""
    return {
        "fidelity_help": FIDELITY_HELP,
        "fidelity_options": FIDELITY_OPTIONS,
        "perceived_help": PERCEIVED_HELP,
        "perceived_options": PERCEIVED_OPTIONS,
        "level_definitions": LEVEL_DEFINITIONS,
        "invariant_note": INVARIANT_NOTE,
    }
