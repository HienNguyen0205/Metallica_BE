"""Tool remember trong vòng agent: đăng ký, không bị gate, phát event.

    PYTHONPATH=. python tests/integration/test_remember_flow.py
"""

import asyncio

from friday import agent, tools
from friday.memory import long_term as lt


def test_remember_is_registered_and_ungated():
    tool = tools.get("remember")
    assert tool is not None, "model không thể gọi thứ không có trong registry"
    assert not tool.needs_confirmation(), "một ghi chú không nên ngắt lời người dùng"


def test_the_loop_records_every_tool_it_runs():
    """Provenance đọc từ đây, nên vòng agent phải ghi lại - không đặc cách tên tool."""
    lt.TURN_TOOLS.set(set())
    lt.mark_tool_used("get_system_metrics")
    lt.mark_tool_used("search_web")
    assert lt.TURN_TOOLS.get() == {"get_system_metrics", "search_web"}
    assert lt.current_provenance() == "tool"


def test_a_write_reaches_the_operator_as_an_event():
    events = []
    lt.clear()
    lt.TURN_TOOLS.set(set())
    lt.store_configured = lambda: True
    lt.store_insert = lambda fact, prov, emb: {
        "id": 1, "fact": fact, "provenance": prov, "use_count": 0, "last_used_at": "2026-01-01",
    }

    async def fake_embed(texts):
        return [[1.0, 0.0] for _ in texts]

    lt.embed = fake_embed

    async def drain():
        async for event in agent.emit_memory_event({"remembered": "x", "id": 1, "provenance": "user"}):
            events.append(event)

    asyncio.run(drain())
    assert events and events[0].kind == "memory", events
    # Ghi âm thầm là ghi không ai kiểm được - đây là toàn bộ biện pháp bảo vệ.
    assert events[0].payload["fact"] == "x"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all remember flow tests passed")
