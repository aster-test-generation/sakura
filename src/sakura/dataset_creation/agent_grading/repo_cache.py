"""Host-side git mirror cache for efficient per-container repo clones.

Ported from sakura-uncertainty (dataset_construction/sandbox/repo_cache.py).
Cloning a repository from its network origin once per container is wasteful
when many containers target the same project. Instead we create a single bare
mirror from the already-present local checkout (which carries the full history
through the submodule's gitdir) and mount it read-only into every container,
which then clones from it locally.

The mirror is keyed by repo name (``<cache_dir>/<repo>.git``) and reused across
runs. Reuse is commit-aware: when a ``required_commit`` is given and an existing
mirror lacks it (for example because the dataset submodule was advanced to a new
pinned commit since the mirror was built), the mirror is refreshed from its
source before being returned, so in-container ``git reset`` never fails on a
missing object.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

CACHE_DIR_NAME = ".repo-cache"


class RepoCacheError(RuntimeError):
    """Raised when a repository mirror cannot be created or refreshed."""


def _git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _has_commit(mirror: Path, commit: str) -> bool:
    return (
        _git(["-C", str(mirror), "cat-file", "-e", f"{commit}^{{commit}}"]).returncode
        == 0
    )


def _refresh_mirror(mirror: Path) -> None:
    result = _git(["-C", str(mirror), "remote", "update", "--prune"])
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RepoCacheError(f"Failed to refresh mirror {mirror}:\n{detail}")


def ensure_repo_mirror(
    *,
    source_checkout: Path,
    cache_dir: Path,
    repo: str,
    required_commit: str | None = None,
) -> Path:
    """Ensure a bare mirror of ``repo`` exists (and contains ``required_commit``).

    Args:
        source_checkout: The local working checkout to mirror (e.g.
            ``resources/datasets/<repo>``). Its full history is resolved through
            the gitdir pointer, so no network access is required.
        cache_dir: Directory holding per-repo bare mirrors.
        repo: Repository name, used to name the mirror directory.
        required_commit: When set, the returned mirror is guaranteed to contain
            this commit (the mirror is refreshed from its source if it does not).

    Returns:
        Path to the ``<repo>.git`` bare mirror.
    """
    if not source_checkout.exists():
        raise RepoCacheError(f"Repo checkout does not exist: {source_checkout}")

    mirror = cache_dir / f"{repo}.git"
    if (mirror / "HEAD").exists():
        if required_commit is None or _has_commit(mirror, required_commit):
            return mirror
        _refresh_mirror(mirror)
        if not _has_commit(mirror, required_commit):
            raise RepoCacheError(
                f"Mirror {mirror} still missing commit {required_commit} after "
                f"refreshing from {source_checkout}."
            )
        return mirror

    cache_dir.mkdir(parents=True, exist_ok=True)
    result = _git(["clone", "--mirror", str(source_checkout), str(mirror)])
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RepoCacheError(
            f"Failed to mirror {source_checkout} into {mirror}:\n{detail}"
        )
    if required_commit is not None and not _has_commit(mirror, required_commit):
        raise RepoCacheError(
            f"Freshly mirrored {source_checkout} does not contain commit "
            f"{required_commit}."
        )
    return mirror
