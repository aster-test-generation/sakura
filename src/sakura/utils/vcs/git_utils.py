from __future__ import annotations

import subprocess
from pathlib import Path

from sakura.utils.pretty.color_logger import RichLog


class GitUtilities:
    """Helpers for verifying and resetting git submodules."""

    @staticmethod
    def has_working_tree_changes(repo_path: Path) -> bool:
        """Return True when the repo has any local changes."""
        repo_path = Path(repo_path)
        GitUtilities._ensure_git_repo(repo_path)
        GitUtilities._clean_lock_files(repo_path)
        result = GitUtilities._run_git_command(
            ["git", "status", "--porcelain"], repo_path
        )
        return bool(result.stdout.strip())

    @staticmethod
    def reset_submodule(repo_path: Path) -> None:
        """Reset a submodule to the pinned commit.

        This discards tracked changes and removes any untracked (and ignored)
        files so each run starts from a clean checkout.
        """
        repo_path = Path(repo_path)
        GitUtilities._ensure_git_repo(repo_path)
        GitUtilities._clean_lock_files(repo_path)
        GitUtilities._run_git_command(["git", "reset", "--hard"], repo_path)
        GitUtilities._run_git_command(["git", "clean", "-fdx"], repo_path)

    @staticmethod
    def reset_submodules_in_dir(submodule_dir: Path) -> None:
        """Reset all submodules located directly under the directory."""
        submodule_dir = Path(submodule_dir)
        if not submodule_dir.exists() or not submodule_dir.is_dir():
            raise ValueError(
                f"Submodule directory {submodule_dir} does not exist or is not a directory."
            )

        for submodule_path in sorted(submodule_dir.iterdir()):
            if not submodule_path.is_dir():
                continue
            if not GitUtilities._is_git_repo(submodule_path):
                continue
            RichLog.info(f"Resetting submodule at {submodule_path}")
            GitUtilities.reset_submodule(submodule_path)

    @staticmethod
    def _get_git_dir(repo_path: Path) -> Path | None:
        """Resolve the actual .git directory for a repo or submodule."""
        git_path = repo_path / ".git"
        if not git_path.exists():
            return None
        if git_path.is_dir():
            return git_path
        # Submodules have a .git file pointing to the actual git directory
        try:
            content = git_path.read_text().strip()
            if content.startswith("gitdir: "):
                gitdir = Path(content[8:])
                if not gitdir.is_absolute():
                    gitdir = (repo_path / gitdir).resolve()
                return gitdir
        except (OSError, IOError):
            pass
        return None

    @staticmethod
    def _clean_lock_files(repo_path: Path) -> None:
        """Remove stale git lock files left by crashed processes."""
        git_dir = GitUtilities._get_git_dir(repo_path)
        if git_dir is None:
            return
        lock_file = git_dir / "index.lock"
        if lock_file.exists():
            RichLog.warn(f"Removing stale git lock file: {lock_file}")
            try:
                lock_file.unlink()
            except OSError as exc:
                RichLog.warn(f"Failed to remove lock file: {exc}")

    @staticmethod
    def _is_git_repo(repo_path: Path) -> bool:
        return (repo_path / ".git").exists()

    @staticmethod
    def _ensure_git_repo(repo_path: Path) -> None:
        if not repo_path.exists() or not repo_path.is_dir():
            raise ValueError(
                f"Repository path {repo_path} does not exist or is not a directory."
            )
        if not GitUtilities._is_git_repo(repo_path):
            raise ValueError(f"Repository path {repo_path} is not a git repository.")

    @staticmethod
    def _run_git_command(
        args: list[str], repo_path: Path
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            args,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip()
            command = " ".join(args)
            message = f"Git command failed in {repo_path}: {command}"
            if details:
                message = f"{message} ({details})"
            raise RuntimeError(message)
        return result
