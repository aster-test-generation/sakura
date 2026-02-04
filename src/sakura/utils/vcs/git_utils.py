from __future__ import annotations
_B='git'
_A=None
import subprocess
from pathlib import Path
from sakura.utils.pretty.color_logger import RichLog
class GitUtilities:
	'Helpers for verifying and resetting git submodules.'
	@staticmethod
	def has_working_tree_changes(repo_path:Path)->bool:'Return True when the repo has any local changes.';A=repo_path;A=Path(A);GitUtilities._ensure_git_repo(A);GitUtilities._clean_lock_files(A);B=GitUtilities._run_git_command([_B,'status','--porcelain'],A);return bool(B.stdout.strip())
	@staticmethod
	def reset_submodule(repo_path:Path)->_A:'Reset a submodule to the pinned commit.\n\n        This discards tracked changes and removes any untracked (and ignored)\n        files so each run starts from a clean checkout.\n        ';A=repo_path;A=Path(A);GitUtilities._ensure_git_repo(A);GitUtilities._clean_lock_files(A);GitUtilities._run_git_command([_B,'reset','--hard'],A);GitUtilities._run_git_command([_B,'clean','-fdx'],A)
	@staticmethod
	def reset_submodules_in_dir(submodule_dir:Path)->_A:
		'Reset all submodules located directly under the directory.';A=submodule_dir;A=Path(A)
		if not A.exists()or not A.is_dir():raise ValueError(f"Submodule directory {A} does not exist or is not a directory.")
		for B in sorted(A.iterdir()):
			if not B.is_dir():continue
			if not GitUtilities._is_git_repo(B):continue
			RichLog.info(f"Resetting submodule at {B}");GitUtilities.reset_submodule(B)
	@staticmethod
	def _get_git_dir(repo_path:Path)->Path|_A:
		'Resolve the actual .git directory for a repo or submodule.';C=repo_path;A=C/'.git'
		if not A.exists():return
		if A.is_dir():return A
		try:
			D=A.read_text().strip()
			if D.startswith('gitdir: '):
				B=Path(D[8:])
				if not B.is_absolute():B=(C/B).resolve()
				return B
		except(OSError,IOError):pass
	@staticmethod
	def _clean_lock_files(repo_path:Path)->_A:
		'Remove stale git lock files left by crashed processes.';B=GitUtilities._get_git_dir(repo_path)
		if B is _A:return
		A=B/'index.lock'
		if A.exists():
			RichLog.warn(f"Removing stale git lock file: {A}")
			try:A.unlink()
			except OSError as C:RichLog.warn(f"Failed to remove lock file: {C}")
	@staticmethod
	def _is_git_repo(repo_path:Path)->bool:return(repo_path/'.git').exists()
	@staticmethod
	def _ensure_git_repo(repo_path:Path)->_A:
		A=repo_path
		if not A.exists()or not A.is_dir():raise ValueError(f"Repository path {A} does not exist or is not a directory.")
		if not GitUtilities._is_git_repo(A):raise ValueError(f"Repository path {A} is not a git repository.")
	@staticmethod
	def _run_git_command(args:list[str],repo_path:Path)->subprocess.CompletedProcess[str]:
		C=repo_path;A=subprocess.run(args,cwd=str(C),capture_output=True,text=True)
		if A.returncode!=0:
			D=A.stderr.strip()or A.stdout.strip();E=' '.join(args);B=f"Git command failed in {C}: {E}"
			if D:B=f"{B} ({D})"
			raise RuntimeError(B)
		return A