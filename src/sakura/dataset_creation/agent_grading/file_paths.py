"""Derive the repo-relative source file for a qualified Java class name.

The derived path is only a *hint* embedded in the agent prompt (the agent is
told to search the checkout if the hint is wrong), so this is deliberately a
cheap filesystem heuristic rather than a full static analysis: strict
package-suffix matching first, then any file named after the outer class,
preferring Maven test roots.
"""

from __future__ import annotations

from pathlib import Path

_EXCLUDED_PARTS = {".git", "target", "build"}


def _outer_class_and_package(qualified_class_name: str) -> tuple[str, list[str]]:
    """Split ``org.foo.Outer$Inner`` / ``org.foo.Outer.Inner`` into file parts.

    The source file is named after the outermost class: the first dot segment
    that starts with an uppercase letter (falling back to the last segment).
    """
    name = qualified_class_name.split("$")[0].strip()
    segments = [segment for segment in name.split(".") if segment]
    if not segments:
        raise ValueError(f"Empty qualified class name: {qualified_class_name!r}")
    class_index = next(
        (i for i, segment in enumerate(segments) if segment[:1].isupper()),
        len(segments) - 1,
    )
    return segments[class_index], segments[:class_index]


def _candidates(project_dir: Path, outer_class: str) -> list[Path]:
    return [
        path
        for path in project_dir.rglob(f"{outer_class}.java")
        if not _EXCLUDED_PARTS.intersection(path.relative_to(project_dir).parts)
    ]


def _pick(candidates: list[Path], project_dir: Path) -> str | None:
    if not candidates:
        return None
    relative = sorted(path.relative_to(project_dir).as_posix() for path in candidates)
    for candidate in relative:
        if "src/test/" in candidate:
            return candidate
    return relative[0]


def derive_java_file_path(
    project_dir: Path, qualified_class_name: str
) -> str | None:
    """Best-effort repo-relative posix path of the class's ``.java`` file.

    Returns None when no plausible file is found; callers pass the miss through
    to the agent, which searches the checkout itself.
    """
    try:
        outer_class, package_parts = _outer_class_and_package(qualified_class_name)
    except ValueError:
        return None

    candidates = _candidates(project_dir, outer_class)
    package_suffix = "/".join([*package_parts, f"{outer_class}.java"])
    exact = [
        path
        for path in candidates
        if path.relative_to(project_dir).as_posix().endswith(package_suffix)
    ]
    return _pick(exact, project_dir) or _pick(candidates, project_dir)
