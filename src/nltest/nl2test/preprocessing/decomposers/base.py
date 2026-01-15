from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar

from nltest.utils.llm import LLMClient

SchemaT = TypeVar("SchemaT")


class BaseDecomposer(ABC):
    @abstractmethod
    def decompose(self, nl_description: str) -> Any:
        raise NotImplementedError

    def invoke_with_retries(
        self,
        *,
        client: LLMClient,
        system_prompt: str,
        base_chat_prompt: str,
        schema: Type[SchemaT],
        strict: bool = True,
        total_attempts: int = 3,
    ) -> SchemaT:
        """Call structured LLM output with retries and feedback about prior failures."""
        failures: list[str] = []
        last_error: Exception | None = None

        for attempt in range(1, total_attempts + 1):
            augmented_chat = base_chat_prompt

            if failures:
                failure_block = "\n".join(failures)
                augmented_chat = (
                    f"{base_chat_prompt}\n\n"
                    "Previous failed attempts while producing structured output:\n"
                    f"{failure_block}\n"
                    "Please fix the issues above and respond with valid structured data."
                )

            try:
                # Should throw exception with strict=True
                return client.invoke_prompts(
                    system=system_prompt,
                    chat=augmented_chat,
                    schema=schema,
                    strict=strict,
                )
            except Exception as exc:
                last_error = exc
                failure_excerpt = (str(exc) or repr(exc)).strip()
                failure_excerpt = failure_excerpt[:800]
                failures.append(f"Attempt {attempt}: {failure_excerpt}")

        if last_error is not None:
            raise last_error
        raise RuntimeError(
            "Structured prompt invocation failed without raising an error."
        )
