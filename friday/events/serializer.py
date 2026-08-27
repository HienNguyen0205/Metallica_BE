"""SSE wire-format serializer — domain event -> SSE frame."""

import json
from typing import Any


def sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def serialize(event: str, payload: dict[str, Any]) -> str:
    """Alias that matches the guide's `Serializer` concept."""
    return sse(event, payload)
