"""Tool remember trong vòng agent: đăng ký, không bị gate, phát event.

    PYTHONPATH=. python tests/integration/test_remember_flow.py
"""

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from friday import agent, tools
from friday.memory import long_term as lt


class _FakeRememberProvider(BaseHTTPRequestHandler):
    """Scripts one `remember` tool call, then a plain answer.

    Same technique as tests/integration/test_provider.py's FakeProvider: a
    throwaway local OpenAI-compatible server so agent.run() runs for real,
    just scripted for the `remember` tool instead of get_system_metrics.
    """

    def log_message(self, *_args):  # keep test output clean
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        already_ran = any(m.get("role") == "tool" for m in body["messages"])
        if already_ran:
            message = {"role": "assistant", "content": "Noted.", "tool_calls": None}
            finish_reason = "stop"
        else:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "remember",
                            "arguments": json.dumps({"fact": "operator prefers dark mode"}),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"

        payload = {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": "fake",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


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
    """Drives the real agent.run() loop — not a stand-in for the emission.

    Ghi âm thầm là ghi không ai kiểm được - đây là toàn bộ biện pháp bảo vệ, nên
    phải test đường thật: model gọi remember qua vòng lặp thật, run_remember
    thật sự ghi, rồi event phải lọt ra ngoài cho operator thấy.
    """
    lt.clear()
    lt.store_configured = lambda: True
    lt.store_insert = lambda fact, prov, emb: {
        "id": 1, "fact": fact, "provenance": prov, "use_count": 0, "last_used_at": "2026-01-01",
    }

    async def fake_embed(texts):
        return [[1.0, 0.0] for _ in texts]

    lt.embed = fake_embed

    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeRememberProvider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    previous = {k: os.environ.get(k) for k in ("FRIDAY_LLM_BASE_URL", "FRIDAY_LLM_API_KEY")}
    os.environ["FRIDAY_LLM_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    os.environ["FRIDAY_LLM_API_KEY"] = "test-key"

    events = []
    try:
        result = agent.AgentResult(text="")

        async def approve(*_):
            raise AssertionError("remember is low risk and must not ask the operator")

        async def drive():
            async for event in agent.run("remember that I like dark mode", approve, result):
                events.append(event)

        asyncio.run(drive())
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        server.shutdown()
        server.server_close()

    memory_events = [e for e in events if e.kind == "memory"]
    assert memory_events, events
    assert memory_events[0].payload["fact"] == "operator prefers dark mode", memory_events
    assert memory_events[0].payload["provenance"] == "user", memory_events


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all remember flow tests passed")
