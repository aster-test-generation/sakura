import re, json, ast
from typing import Type, TypeVar, get_origin, get_args, List, Union

T = TypeVar("T")


class FormatValidator:
    @staticmethod
    def _strip_json_fences(raw: str) -> str:
        """Remove ```json ...``` or ```...``` fences."""
        return re.sub(r'```(?:json)?\s*([\s\S]*?)\s*```', r'\1', raw).strip()

    @staticmethod
    def _tuples_to_lists(obj):
        """Recursively convert tuples from ast.literal_eval to lists so that the resulting data structure is JSON‑serializable."""
        if isinstance(obj, tuple):
            return [FormatValidator._tuples_to_lists(x) for x in obj]
        if isinstance(obj, list):
            return [FormatValidator._tuples_to_lists(x) for x in obj]
        if isinstance(obj, dict):
            return {k: FormatValidator._tuples_to_lists(v) for k, v in obj.items()}
        return obj

    @staticmethod
    def strip_java_block(raw: str) -> str | None:
        pattern = r'```java\n(.*?)\n```'
        matches = re.findall(pattern, raw, re.DOTALL)
        return matches[0] if matches else None

    @staticmethod
    def validate(text: str, output_type: Type[T]) -> Union[T, List[T]]:
        cleaned = FormatValidator._strip_json_fences(text)

        # First try strict JSON
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as je:
            # Fall back to python‑literal syntax for tuples, single quotes, etc.
            try:
                parsed = ast.literal_eval(cleaned)
                parsed = FormatValidator._tuples_to_lists(parsed)
            except (ValueError, SyntaxError):
                # Last‑chance list wrapper fallback
                m = re.fullmatch(r'\s*(\[[\s\S]*\])\s*', cleaned)
                if m:
                    parsed = json.loads(m.group(1))
                else:
                    raise ValueError(
                        f"Failed to parse output as JSON or Python literal. "
                        f"json error was: {je}"
                    ) from je

        origin = get_origin(output_type)
        if origin in (list, List):
            elem_t = get_args(output_type)[0]
            if not isinstance(parsed, list):
                raise ValueError(f"Expected a JSON array, got {type(parsed).__name__}")
            return [elem_t(**item) for item in parsed]

        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")

        return output_type(**parsed)
