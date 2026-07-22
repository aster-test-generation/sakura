"""Host-side driver for the agent-grading Docker sandbox.

Ported and trimmed from sakura-uncertainty
(dataset_construction/sandbox/docker_runner.py): builds the image once, then
runs one hardened container per description around a JSON spec mounted under
``/out`` with a read-only repository mirror at ``/repo-src``.

The image build context is this package directory itself (not the sakura repo
root), so the container ships only the self-contained ``agent_grading``
modules and never the full sakura dependency set.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_IMAGE = "sakura-agent-grading:latest"
DEFAULT_DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.agent-grading"
DEFAULT_NETWORK = "bridge"
CONTAINER_OUTPUT_ROOT = "/out"
CONTAINER_REPO_SOURCE = "/repo-src"
DEFAULT_AUTH_ENV_NAMES = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
)
DEFAULT_PASSTHROUGH_ENV_NAMES = (
    *DEFAULT_AUTH_ENV_NAMES,
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_CUSTOM_MODEL_OPTION",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS",
)


class DockerSandboxError(RuntimeError):
    """Raised when the Docker sandbox cannot be built or run."""


class DockerSandboxRunner:
    """Builds the sandbox image once and runs one container per spec."""

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        dockerfile: Path = DEFAULT_DOCKERFILE,
        network: str = DEFAULT_NETWORK,
        memory: str | None = "6g",
        cpus: float | None = None,
        pids_limit: int = 1024,
        env_names: Iterable[str] = DEFAULT_PASSTHROUGH_ENV_NAMES,
        require_auth_env: bool = True,
    ) -> None:
        self.image = image
        self.dockerfile = dockerfile
        self.network = network
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.env_names = tuple(dict.fromkeys(env_names))
        self.require_auth_env = require_auth_env

    def ensure_auth(self) -> None:
        """Fail fast if auth is required but no Anthropic auth env var is set."""
        if self.require_auth_env and not any(
            os.environ.get(name) for name in DEFAULT_AUTH_ENV_NAMES
        ):
            raise DockerSandboxError(
                "No Anthropic auth environment variable is set. Set one of "
                f"{', '.join(DEFAULT_AUTH_ENV_NAMES)} or pass --allow-missing-auth."
            )

    def build_image(self) -> None:
        """Build the sandbox image once. Reused by every subsequent container."""
        if not self.dockerfile.exists():
            raise DockerSandboxError(f"Dockerfile does not exist: {self.dockerfile}")
        command = [
            "docker",
            "build",
            "-f",
            str(self.dockerfile),
            "-t",
            self.image,
            str(PACKAGE_DIR),
        ]
        self._run(command, cwd=REPO_ROOT)

    def run_spec(
        self,
        *,
        spec: dict[str, Any],
        output_root: Path,
        spec_path: Path,
        repo_source: Path,
    ) -> None:
        """Run a single container around ``spec``.

        Args:
            spec: The JSON-serializable task spec handed to the container.
            output_root: Host directory mounted read-write at ``/out``.
            spec_path: Where to write the spec on the host (must live under
                ``output_root`` so the container can read it from ``/out``).
            repo_source: Host path mounted read-only at ``/repo-src`` (a local
                git mirror the container clones from, avoiding network fetches).
        """
        output_root.mkdir(parents=True, exist_ok=True)
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        container_spec_path = self._container_path(spec_path, output_root)
        command = self._docker_command(
            output_root=output_root,
            container_spec_path=container_spec_path,
            repo_source=repo_source,
        )
        self._run(command, cwd=REPO_ROOT)

    @staticmethod
    def _container_path(spec_path: Path, output_root: Path) -> str:
        relative = spec_path.resolve().relative_to(output_root.resolve())
        return f"{CONTAINER_OUTPUT_ROOT}/{relative.as_posix()}"

    def _docker_command(
        self,
        *,
        output_root: Path,
        container_spec_path: str,
        repo_source: Path,
    ) -> list[str]:
        command = [
            "docker",
            "run",
            "--rm",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--network",
            self.network,
            "--pids-limit",
            str(self.pids_limit),
        ]
        if self.memory:
            command.extend(["--memory", self.memory])
        if self.cpus is not None:
            command.extend(["--cpus", str(self.cpus)])

        for env_name in self.env_names:
            if os.environ.get(env_name):
                command.extend(["-e", env_name])

        command.extend(
            ["-v", f"{output_root.resolve()}:{CONTAINER_OUTPUT_ROOT}:rw"]
        )
        command.extend(
            ["-v", f"{repo_source.resolve()}:{CONTAINER_REPO_SOURCE}:ro"]
        )

        command.extend([self.image, "--task-spec", container_spec_path])
        return command

    @staticmethod
    def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(command, cwd=cwd, text=True, check=False)
        if result.returncode != 0:
            raise DockerSandboxError(
                f"Command failed with exit code {result.returncode}: "
                + " ".join(command)
            )
        return result
