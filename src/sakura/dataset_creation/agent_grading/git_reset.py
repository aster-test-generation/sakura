"""Submodule-aware git reset helpers.

Ported from sakura-uncertainty (dataset_construction/git_vcs/git_reset.py):
resolve a checkout's pinned commit (including the pinned commit recorded by a
superproject for a submodule) and hard-reset back to it. Used by the host to
resolve the pinned commit and by the container to freshen its clone.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


class GitVcsError(RuntimeError):
    pass


@dataclass(frozen=True)
class GitResetTarget:
    pinned_commit: str


def _run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        command = " ".join(["git", "-C", str(cwd), *args])
        detail = result.stderr.strip() or result.stdout.strip()
        raise GitVcsError(f"Git command failed: {command}\n{detail}")
    return result


def _git_output(args: list[str], *, cwd: Path) -> str:
    return _run_git(args, cwd=cwd).stdout.strip()


def _repo_root(directory: Path) -> Path:
    if not directory.exists():
        raise GitVcsError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise GitVcsError(f"Path is not a directory: {directory}")
    return Path(_git_output(["rev-parse", "--show-toplevel"], cwd=directory)).resolve()


def _superproject_root(repo_root: Path) -> Path | None:
    output = _git_output(
        ["rev-parse", "--show-superproject-working-tree"],
        cwd=repo_root,
    )
    if not output:
        return None
    return Path(output).resolve()


def _superproject_relative_path(repo_root: Path, superproject_root: Path) -> str:
    return os.path.relpath(repo_root, superproject_root).replace(os.sep, "/")


def _pinned_submodule_commit(
    *,
    repo_root: Path,
    superproject_root: Path,
) -> str:
    relative_path = _superproject_relative_path(repo_root, superproject_root)
    stage_output = _git_output(
        ["ls-files", "--stage", "--", relative_path],
        cwd=superproject_root,
    )
    for line in stage_output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "160000":
            return parts[1]
    raise GitVcsError(
        f"Could not find pinned submodule commit for {repo_root} "
        f"in {superproject_root}"
    )


def _reset_nested_submodules(repo_root: Path) -> None:
    status = _run_git(["submodule", "status", "--recursive"], cwd=repo_root)
    if not status.stdout.strip():
        return

    _run_git(
        ["submodule", "foreach", "--recursive", "git reset --hard && git clean -ffdx"],
        cwd=repo_root,
    )
    _run_git(["submodule", "update", "--init", "--recursive", "--force"], cwd=repo_root)
    _run_git(
        ["submodule", "foreach", "--recursive", "git reset --hard && git clean -ffdx"],
        cwd=repo_root,
    )


def resolve_reset_target(directory: str | Path) -> GitResetTarget:
    target_dir = Path(directory).expanduser().resolve()
    repo_root = _repo_root(target_dir)
    superproject_root = _superproject_root(repo_root)

    if superproject_root is None:
        pinned_commit = _git_output(["rev-parse", "HEAD"], cwd=repo_root)
        return GitResetTarget(pinned_commit=pinned_commit)

    pinned_commit = _pinned_submodule_commit(
        repo_root=repo_root,
        superproject_root=superproject_root,
    )
    return GitResetTarget(pinned_commit=pinned_commit)


def reset_to_commit(directory: str | Path, commit: str) -> None:
    target_dir = Path(directory).expanduser().resolve()
    repo_root = _repo_root(target_dir)

    _run_git(["cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo_root)
    _run_git(["reset", "--hard", commit], cwd=repo_root)
    _run_git(["clean", "-ffdx"], cwd=repo_root)
    _reset_nested_submodules(repo_root)
