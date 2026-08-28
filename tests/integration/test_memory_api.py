"""Đường xem lại và xoá. Trong thiết kế này nó là lớp bảo vệ duy nhất.

    PYTHONPATH=. python tests/integration/test_memory_api.py
"""

import os

os.environ["FRIDAY_ALLOWED_ORIGINS"] = "http://localhost:3000"

from fastapi.testclient import TestClient

from friday.main import app
from friday.memory import long_term as lt

client = TestClient(app)


def seed():
    lt.clear()
    lt.CACHE.append(lt.Memory(id=1, fact="thích đơn vị mét", provenance="user", embedding=[1.0]))
    lt.CACHE.append(lt.Memory(id=2, fact="đọc trên mạng", provenance="tool", embedding=[1.0]))


def test_listing_shows_provenance_so_web_sourced_facts_are_visible():
    seed()
    body = client.get("/memory").json()
    kinds = {m["id"]: m["provenance"] for m in body["memories"]}
    assert kinds == {1: "user", 2: "tool"}, body
    # Vector không bao giờ ra ngoài: 768 số float không giúp ai đọc, chỉ làm
    # response phình lên.
    assert "embedding" not in body["memories"][0]


def test_deleting_removes_it():
    seed()
    lt.store_delete = lambda memory_id: None
    assert client.delete("/memory/1").json()["ok"] is True
    assert [m.id for m in lt.CACHE] == [2]


def test_deleting_something_that_is_not_there_is_not_a_success():
    seed()
    lt.store_delete = lambda memory_id: None
    assert client.delete("/memory/999").json()["ok"] is False


def test_a_foreign_origin_cannot_read_or_erase_what_friday_knows():
    seed()
    headers = {"origin": "https://evil.example"}
    assert client.get("/memory", headers=headers).status_code == 403
    assert client.delete("/memory/1", headers=headers).status_code == 403


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all memory API tests passed")
