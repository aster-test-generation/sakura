import re
from typing import Optional


class FormatValidator:
    """Utility helpers for massaging formatter-related LLM responses."""

    _FENCE_LANGUAGE_HINTS = {
        "java", "javascript", "js", "typescript", "ts", "python", "py",
        "c", "c++", "cpp", "csharp", "c#", "cs",
        "go", "golang", "kotlin", "swift", "scala", "groovy",
        "ruby", "php", "bash", "sh", "shell", "powershell", "ps", "ps1",
        "sql", "json", "yaml", "yml", "xml", "html", "css",
        "text", "plain", "plaintext", "markdown", "md",
    }

    @staticmethod
    def sanitize_code_block(code: str) -> str:
        """
        Strip markdown or quote wrappers like ```java ... ``` or ''' ... '''.
        """
        if not isinstance(code, str):
            return code

        stripped = code.strip()
        if not stripped:
            return stripped

        # Fenced blocks: ``` ... ``` or ~~~ ... ~~~
        for fence in ("```", "~~~"):
            sanitized = FormatValidator._strip_fenced_block(stripped, fence)
            if sanitized is not None:
                return sanitized

        # Simple quote wrappers: '''...''' or """..."""
        for quote in ("'''", '"""'):
            sanitized = FormatValidator._strip_simple_wrapper(stripped, quote)
            if sanitized is not None:
                return sanitized

        return stripped

    @staticmethod
    def _strip_fenced_block(text: str, fence: str) -> Optional[str]:
        if not text.startswith(fence):
            return None

        last = text.rfind(fence)
        if last <= len(fence):
            return None

        inner = text[len(fence):last]
        inner = FormatValidator._strip_language_hint(inner)
        return inner.strip()

    @staticmethod
    def _strip_language_hint(inner: str) -> str:
        # Drop leading CRs (Windows newlines)
        trimmed = inner.lstrip("\r")

        # Plain fenced block with immediate newline:
        # ```\ncode\n``` -> "code\n"
        if trimmed.startswith("\n"):
            return trimmed.lstrip("\r\n")

        # Allow and ignore leading horizontal whitespace before a language tag:
        # ```   java\ncode``` or ``` java code```
        candidate = trimmed.lstrip(" \t")

        # Look for "<token><space/newline>" at the very start
        match = re.match(r"([A-Za-z0-9_+\-#.]+)([\t ]+|\r?\n)", candidate)
        if match and FormatValidator._looks_like_language_hint(match.group(1)):
            # Strip the language token + following whitespace/newline
            return candidate[match.end():]

        # No language tag detected; keep original (with leading spaces)
        return trimmed

    @staticmethod
    def _looks_like_language_hint(token: str) -> bool:
        normalized = token.strip().lower()
        if not normalized:
            return False
        normalized = normalized.replace("language-", "")
        return normalized in FormatValidator._FENCE_LANGUAGE_HINTS

    @staticmethod
    def _strip_simple_wrapper(text: str, wrapper: str) -> Optional[str]:
        if text.startswith(wrapper) and text.endswith(wrapper):
            inner = text[len(wrapper):-len(wrapper)]
            inner = FormatValidator._strip_language_hint(inner)
            return inner.strip()
        return None
