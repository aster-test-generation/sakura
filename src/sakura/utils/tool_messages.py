from __future__ import annotations

import json
from typing import Any, Dict, Mapping


def build_tool_ok(data: Any) -> Dict[str, Any]:
    """Construct a standard success envelope for tool outputs."""
    return {"status": "ok", "data": to_json_safe(data)}


def build_tool_error(
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Construct a standard error envelope for tool outputs."""
    return {
        "status": "error",
        "error": {
            "code": code,
            "message": message,
            "details": to_json_safe(dict(details) if details is not None else {}),
        },
    }


def dumps_envelope(envelope: Dict[str, Any]) -> str:
    """Serialize an envelope with consistent JSON formatting."""
    return json.dumps(envelope, separators=(",", ":"), ensure_ascii=False)


def format_tool_ok(data: Any) -> str:
    """Serialize a success envelope."""
    return dumps_envelope(build_tool_ok(data))


def format_tool_error(
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
) -> str:
    """Serialize an error envelope."""
    return dumps_envelope(build_tool_error(code, message, details=details))


def to_json_safe(value: Any) -> Any:
    """Convert values to JSON-serializable structures, falling back to string."""
    if hasattr(value, "model_dump"):
        try:
            return to_json_safe(value.model_dump())
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        sequence = [to_json_safe(v) for v in value]
        return sequence if isinstance(value, list) else sequence
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
