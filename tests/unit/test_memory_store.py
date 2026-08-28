"""store.py: dựng request PostgREST đúng, và hỏng thì hỏng ra StoreError.

    PYTHONPATH=. python tests/unit/test_memory_store.py
"""

import json
import os
import urllib.error

os.environ["SUPABASE_URL"] = "https://proj.supabase.co"
os.environ["SUPABASE_SERVICE_KEY"] = "service-key-123"

from friday.memory import store


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def capture(payload, sink):
    def fake_urlopen(req, timeout=None):
        sink.append(req)
        return FakeResponse(payload)

    return fake_urlopen


def test_credentials_ride_on_every_request():
    sink = []
    store.urlopen = capture([], sink)
    store.select_all()
    req = sink[0]
    assert req.headers["Apikey"] == "service-key-123"
    assert req.headers["Authorization"] == "Bearer service-key-123"


def test_select_asks_for_every_column_the_cache_needs():
    sink = []
    store.urlopen = capture([], sink)
    store.select_all()
    url = sink[0].full_url
    for column in ("id", "fact", "provenance", "embedding", "last_used_at", "use_count"):
        assert column in url, f"{column} missing from {url}"


def test_insert_sends_the_vector_as_a_postgrest_literal():
    sink = []
    store.urlopen = capture([{"id": 7}], sink)
    row = store.insert("con thích đơn vị mét", "user", [0.5, 0.5])
    body = json.loads(sink[0].data.decode())
    # pgvector nhận chuỗi "[a,b]", không phải mảng JSON — gửi mảy thì Postgres
    # từ chối với một lỗi kiểu khó lần.
    assert body["embedding"] == "[0.5,0.5]", body["embedding"]
    assert row["id"] == 7


def test_delete_filters_by_id_not_by_everything():
    sink = []
    store.urlopen = capture([], sink)
    store.delete(7)
    assert "id=eq.7" in sink[0].full_url, sink[0].full_url
    assert sink[0].get_method() == "DELETE"


def test_a_dead_supabase_raises_storeerror_not_urlerror():
    def boom(req, timeout=None):
        raise urllib.error.URLError("no route to host")

    store.urlopen = boom
    for call in (store.select_all, lambda: store.insert("x", "user", [1.0]), lambda: store.delete(1)):
        try:
            call()
        except store.StoreError:
            pass
        else:
            raise AssertionError(f"{call} swallowed a dead backend")


def test_unconfigured_is_not_an_error_it_is_a_mode():
    saved = os.environ.pop("SUPABASE_URL")
    try:
        assert store.configured() is False
    finally:
        os.environ["SUPABASE_URL"] = saved
    assert store.configured() is True


def test_select_all_parses_vector_from_postgrest_string_format():
    """PostgREST returns pgvector as string "[a,b,c]" not as JSON array."""
    sink = []
    payload = [
        {"id": 1, "fact": "test", "provenance": "user", "embedding": "[0.1,0.2,0.3]", "created_at": "2026-01-01", "last_used_at": "2026-01-01", "use_count": 1},
        {"id": 2, "fact": "test2", "provenance": "user", "embedding": "[0.5,0.6]", "created_at": "2026-01-01", "last_used_at": "2026-01-01", "use_count": 1},
    ]
    store.urlopen = capture(payload, sink)
    rows = store.select_all()
    assert rows[0]["embedding"] == [0.1, 0.2, 0.3], f"got {rows[0]['embedding']}"
    assert rows[1]["embedding"] == [0.5, 0.6], f"got {rows[1]['embedding']}"


def test_select_all_parses_empty_vector_string():
    """Empty vector string "[]" should parse to empty list."""
    sink = []
    payload = [
        {"id": 1, "fact": "test", "provenance": "user", "embedding": "[]", "created_at": "2026-01-01", "last_used_at": "2026-01-01", "use_count": 1},
    ]
    store.urlopen = capture(payload, sink)
    rows = store.select_all()
    assert rows[0]["embedding"] == [], f"got {rows[0]['embedding']}"


def test_touch_does_not_raise_when_backend_dead():
    """touch() swallows StoreError and logs instead."""
    def boom(req, timeout=None):
        raise urllib.error.URLError("no route to host")

    store.urlopen = boom
    store.touch([1, 2, 3])  # Should not raise


def test_touch_empty_list_makes_no_request():
    """touch() with empty id list should return early without making a request."""
    sink = []
    store.urlopen = capture([], sink)
    store.touch([])
    assert len(sink) == 0, f"expected no requests, got {len(sink)}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all store tests passed")
