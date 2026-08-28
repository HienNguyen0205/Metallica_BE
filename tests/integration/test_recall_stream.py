"""Recall chèn vào prompt, event memory ra tới stream, và cả hai đường hỏng.

    PYTHONPATH=. python tests/integration/test_recall_stream.py
"""

import asyncio
import json

from friday import agent, main
from friday.memory import embed as embed_mod
from friday.memory import long_term as lt
from friday.schema import VisualizationPlan, VizData

PLAN = VisualizationPlan(
    type="radial_gauge", title="X", data=VizData(metrics=[]), answer="a",
)


def parse(chunk):
    name, _, data = chunk.strip().partition("\n")
    return name.removeprefix("event: "), json.loads(data.removeprefix("data: "))


def collect(query="q"):
    async def drain():
        return [parse(c) async for c in main.run_query(query)]

    return asyncio.run(drain())


def stub_planner():
    async def fake_plan(q, a, evidence, pinned_type=None):
        return PLAN

    main.plan = fake_plan


def test_a_relevant_memory_reaches_the_system_prompt():
    lt.clear()
    lt.CACHE.append(lt.Memory(id=1, fact="thích đơn vị mét", provenance="user", embedding=[1.0, 0.0]))

    async def fake_embed(texts):
        return [[1.0, 0.0] for _ in texts]

    embed_mod.embed = fake_embed
    block = asyncio.run(main.recall_block("đo bằng gì"))
    assert "thích đơn vị mét" in block
    assert "<remembered_facts>" in block


def test_a_dead_embedder_skips_recall_without_failing_the_turn():
    lt.clear()
    lt.CACHE.append(lt.Memory(id=1, fact="f", provenance="user", embedding=[1.0, 0.0]))

    async def boom(texts):
        raise embed_mod.EmbedError("provider down")

    embed_mod.embed = boom
    assert asyncio.run(main.recall_block("q")) == "", "recall hỏng không được kéo turn theo"


def test_the_memory_event_reaches_the_stream():
    stub_planner()

    async def fake_agent(query, approve, result, history=(), memories=""):
        yield agent.AgentEvent("memory", {"id": 1, "fact": "đã học", "provenance": "user"})
        result.text = "xong"

    original, agent.run = agent.run, fake_agent
    try:
        events = collect()
    finally:
        agent.run = original

    names = [n for n, _ in events]
    assert "memory" in names, names
    payload = next(p for n, p in events if n == "memory")
    assert payload["fact"] == "đã học" and payload["provenance"] == "user"


def test_memory_is_in_the_declared_event_contract():
    with open("contracts/events.json", encoding="utf-8") as fh:
        contract = json.load(fh)
    # Frontend đọc file này để biết event nào tồn tại. Thêm event mà quên
    # contract là cách để hai bên lệch nhau trong im lặng.
    assert "memory" in contract["events"], contract["events"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all recall stream tests passed")
