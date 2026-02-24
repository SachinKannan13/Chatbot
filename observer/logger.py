from __future__ import annotations

from typing import Any


def _to_json_safe(value: Any) -> Any:
    """Best-effort conversion to JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in list(value)]
    return repr(value)


def summarize_supabase_response(response: Any) -> dict[str, Any]:
    """
    Build a passive, structured snapshot from a Supabase/PostgREST response.
    """
    response_type = type(response).__name__

    raw_data = getattr(response, "data", response)
    serializable = _to_json_safe(raw_data)

    row_count = 0
    if isinstance(raw_data, list):
        row_count = len(raw_data)
    elif isinstance(raw_data, dict):
        row_count = 1

    return {
        "raw_response": serializable,
        "response_type": response_type,
        "row_count": row_count,
    }

