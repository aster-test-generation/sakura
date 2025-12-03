from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OptimizedPrompts:
    """Container for optimized prompt strings."""

    system_prompt: str
    chat_prompt: str


class PromptFormatter:
    """Optimizes prompts using Vertex AI's zero-shot prompt optimizer.

    GCP Setup Requirements:
        1. Enable API: gcloud services enable aiplatform.googleapis.com --project=PROJECT
        2. Authenticate: gcloud auth application-default login
           (or set GOOGLE_APPLICATION_CREDENTIALS to a service account key path)
        3. IAM: Ensure roles/aiplatform.user is granted to the authenticated identity
    """

    def __init__(
        self,
        project: str,
        location: str = "us-central1",
    ):
        """
        Initialize the PromptFormatter.

        Args:
            project: GCP project ID.
            location: GCP region (default: us-central1).
        """
        self._project = project
        self._location = location
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy-initialize the Vertex AI client."""
        if self._client is None:
            import vertexai

            self._client = vertexai.Client(
                project=self._project,
                location=self._location,
            )
        return self._client

    def optimize(
        self,
        system_prompt: str,
        chat_prompt: str,
    ) -> OptimizedPrompts:
        """
        Optimize the given prompts using Vertex AI zero-shot optimizer.

        Args:
            system_prompt: The system instruction prompt.
            chat_prompt: The user/chat prompt.

        Returns:
            OptimizedPrompts containing the optimized strings.
        """
        optimized_system = self._optimize_single(system_prompt)
        optimized_chat = self._optimize_single(chat_prompt)
        return OptimizedPrompts(
            system_prompt=optimized_system,
            chat_prompt=optimized_chat,
        )

    def _optimize_single(self, prompt: str) -> str:
        """Optimize a single prompt string."""
        response = self.client.prompt_optimizer.optimize_prompt(prompt=prompt)
        return response.suggested_prompt
