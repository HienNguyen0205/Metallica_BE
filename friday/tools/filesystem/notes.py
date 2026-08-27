"""Filesystem note tool — high-risk, confined to notes/."""

import re
from pathlib import Path
from typing import Any

NOTES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "notes"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]")


async def run_write_note(payload: dict[str, Any]) -> dict[str, Any]:
    raw = str(payload.get("name", "")).strip()
    body = str(payload.get("body", ""))

    stem = _SAFE_NAME.sub("-", raw)[:64].strip("-")
    if not stem:
        return {"error": "invalid note name"}

    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    target = NOTES_DIR / f"{stem}.md"
    target.write_text(body, encoding="utf-8")
    return {"written": target.name, "bytes": len(body.encode("utf-8"))}
