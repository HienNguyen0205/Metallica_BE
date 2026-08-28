"""Ghi/nạp ký ức, provenance theo turn, và mọi đường hỏng.

    PYTHONPATH=. python tests/unit/test_long_term_io.py
"""

import asyncio

from friday.memory import embed as embed_mod
from friday.memory import long_term as lt
from friday.memory import store


def fake_embed(vectors):
    async def _embed(texts):
        return [vectors[i % len(vectors)] for i in range(len(texts))]

    return _embed


def stub(*, rows=None, insert_row=None, fail=None):
    lt.clear()
    embed_mod_embed = fake_embed([[1.0, 0.0]])
    lt.embed = embed_mod_embed

    def _select_all():
        if fail == "select":
            raise store.StoreError("down")
        return rows or []

    def _insert(fact, provenance, embedding):
        if fail == "insert":
            raise store.StoreError("down")
        return insert_row or {"id": 1, "fact": fact, "provenance": provenance, "use_count": 0, "last_used_at": "2026-01-01"}

    def _delete(memory_id):
        if fail == "delete":
            raise store.StoreError("down")

    lt.store_select_all = _select_all
    lt.store_insert = _insert
    lt.store_delete = _delete
    lt.store_configured = lambda: True


def test_load_fills_the_cache():
    stub(rows=[{"id": 3, "fact": "f", "provenance": "user", "embedding": [1.0, 0.0], "use_count": 0, "last_used_at": "2026-01-01"}])
    assert asyncio.run(lt.load()) == 1
    assert lt.CACHE[0].id == 3


def test_a_dead_store_at_startup_is_a_warning_not_a_crash():
    stub(fail="select")
    assert asyncio.run(lt.load()) == 0, "khởi động phải sống sót qua Supabase chết"
    assert lt.CACHE == []


def test_remember_writes_and_lands_in_the_cache():
    stub()
    out = asyncio.run(lt.run_remember({"fact": "thích đơn vị mét"}))
    assert out["remembered"] == "thích đơn vị mét", out
    assert len(lt.CACHE) == 1


def test_an_empty_fact_is_refused_before_any_call_goes_out():
    stub()
    assert "error" in asyncio.run(lt.run_remember({"fact": "   "}))
    assert lt.CACHE == []


def test_a_long_fact_is_trimmed():
    stub()
    asyncio.run(lt.run_remember({"fact": "x" * (lt.MAX_FACT_CHARS + 100)}))
    assert len(lt.CACHE[0].fact) == lt.MAX_FACT_CHARS


def test_a_failed_write_tells_the_model_instead_of_killing_the_turn():
    stub(fail="insert")
    out = asyncio.run(lt.run_remember({"fact": "gì đó"}))
    assert "error" in out, out
    assert lt.CACHE == [], "ghi hỏng mà cache vẫn nhận là cache nói dối"


def test_provenance_follows_what_ran_this_turn():
    stub()
    lt.TURN_TOOLS.set(set())
    assert lt.current_provenance() == "user"
    lt.mark_tool_used("get_system_metrics")
    assert lt.current_provenance() == "user", "chỉ search_web mới đưa chữ người lạ vào"
    lt.mark_tool_used("search_web")
    assert lt.current_provenance() == "tool"


def test_remember_tags_provenance_from_the_turn():
    stub()
    lt.TURN_TOOLS.set({"search_web"})
    asyncio.run(lt.run_remember({"fact": "đọc trên mạng"}))
    assert lt.CACHE[0].provenance == "tool"


def test_forget_removes_it_from_both_places():
    stub(rows=[{"id": 3, "fact": "f", "provenance": "user", "embedding": [1.0, 0.0], "use_count": 0, "last_used_at": "2026-01-01"}])
    asyncio.run(lt.load())
    assert lt.forget(3) is True
    assert lt.CACHE == []
    assert lt.forget(999) is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all long-term IO tests passed")
