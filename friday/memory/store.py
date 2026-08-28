"""Kho bền cho ký ức dài hạn — Supabase qua PostgREST.

`urllib.request` chứ không phải `httpx`: `httpx` không phải dependency khai báo
của service này, nó chỉ theo `openai` vào, và tự nhận một dependency bắc cầu là
cách để một bản nâng cấp của thư viện khác làm hỏng chỗ này.

Đây là chỗ duy nhất trong package chạm mạng, nên test thay nguyên `urlopen` và
không bao giờ đi ra ngoài.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any
from urllib.request import urlopen  # noqa: F401 — test thay thẳng tên này

log = logging.getLogger("friday.memory")

TABLE = "friday_memory"
COLUMNS = "id,fact,provenance,embedding,created_at,last_used_at,use_count"
TIMEOUT_S = 10.0


class StoreError(RuntimeError):
    """Supabase không trả lời được. Người gọi phải chạy tiếp mà không có ký ức."""


def _base() -> str:
    return (os.getenv("SUPABASE_URL") or "").rstrip("/")


def _key() -> str:
    return os.getenv("SUPABASE_SERVICE_KEY") or ""


def configured() -> bool:
    return bool(_base() and _key())


def _request(method: str, path: str, body: Any = None, prefer: str | None = None) -> Any:
    if not configured():
        raise StoreError("supabase is not configured")

    headers = {
        "apikey": _key(),
        "authorization": f"Bearer {_key()}",
        "content-type": "application/json",
    }
    if prefer:
        headers["prefer"] = prefer

    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{_base()}/rest/v1/{path}", data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=TIMEOUT_S) as resp:
            raw = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as err:
        raise StoreError(f"{method} {path} failed: {err}") from err

    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as err:
        raise StoreError(f"{method} {path} returned non-JSON") from err


def _vector_literal(embedding: list[float]) -> str:
    # pgvector nhận "[a,b,c]" chứ không nhận mảng JSON. Gửi mảng thì Postgres từ
    # chối bằng một lỗi ép kiểu chẳng nhắc gì tới vector.
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def _parse_vector(raw: Any) -> list[float]:
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, str):
        return [float(x) for x in raw.strip("[]").split(",") if x.strip()]
    raise StoreError(f"unreadable embedding of type {type(raw).__name__}")


def select_all() -> list[dict[str, Any]]:
    rows = _request("GET", f"{TABLE}?select={COLUMNS}") or []
    for row in rows:
        row["embedding"] = _parse_vector(row["embedding"])
    return rows


def insert(fact: str, provenance: str, embedding: list[float]) -> dict[str, Any]:
    body = {"fact": fact, "provenance": provenance, "embedding": _vector_literal(embedding)}
    rows = _request("POST", TABLE, body, prefer="return=representation") or []
    if not rows:
        raise StoreError("insert returned no row")
    row = rows[0]
    row["embedding"] = list(embedding)
    return row


def delete(memory_id: int) -> None:
    _request("DELETE", f"{TABLE}?id=eq.{int(memory_id)}")


def touch(ids: list[int]) -> None:
    """Số liệu, không phải dữ liệu — hỏng thì log rồi thôi."""
    if not ids:
        return
    joined = ",".join(str(int(i)) for i in ids)
    try:
        _request("PATCH", f"{TABLE}?id=in.({joined})", {"last_used_at": "now()"})
    except StoreError:
        log.warning("could not update memory usage counters", exc_info=True)
