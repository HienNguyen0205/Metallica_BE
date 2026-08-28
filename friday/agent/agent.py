"""§9 agent loop — plan, call tools, gather results."""

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from friday import llm
from friday.memory import long_term
from friday.tools.registry import REGISTRY, api_tools

from .state import AgentEvent, AgentResult

log = logging.getLogger("friday.agent")

MAX_TURNS = 6

SYSTEM = """You are FRIDAY, an AI operations interface.

Use your tools to answer from real data rather than estimating. \
`get_system_metrics` reads the machine this orchestrator runs on — call it \
whenever the question touches system load, CPU, memory or disk. Do not answer \
those questions from memory or guesswork; you have no numbers until a tool \
gives you some.

`search_web` reaches the public internet, so it can only answer what the public \
internet knows. The public internet knows nothing whatsoever about this machine \
— not its topology, its services, its history, its configuration, nor anything \
else about it, no matter how the question is phrased. Never search for anything \
about this host: either a tool here measures it, or you say you cannot see it. \
Do not search for facts that do not change either.

If a tool is denied by the operator, say so plainly; do not retry it and do not \
substitute invented figures for the data you were refused.

When you have what you need, answer in one or two calm, factual sentences."""

Approver = Callable[[str, str, dict[str, Any]], Awaitable[bool]]


def _echo(call: Any) -> dict[str, Any]:
    return {key: value for key, value in call.model_dump().items() if value is not None}


def _arguments(call: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        log.warning("unparseable arguments for %s", call.function.name)
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def run(
    query: str,
    approve: Approver,
    result: AgentResult,
    history: Sequence[dict[str, str]] = (),
) -> AsyncIterator[AgentEvent]:
    api = llm.client()
    # Provenance của ký ức đọc từ đây. Set mới mỗi turn để hai query song song
    # không thấy tool của nhau.
    long_term.TURN_TOOLS.set(set())
    # §15 — prior exchanges sit between the system prompt and the new question,
    # which is what lets "and the disk?" resolve to anything.
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        *history,
        {"role": "user", "content": query},
    ]

    for _ in range(MAX_TURNS):
        response = await api.chat.completions.create(
            model=llm.model(),
            messages=messages,  # type: ignore[arg-type]
            tools=api_tools(),  # type: ignore[arg-type]
        )
        message = response.choices[0].message
        calls = message.tool_calls or []

        if not calls:
            result.text = (message.content or "").strip()
            return

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [_echo(c) for c in calls],
            }
        )

        for call in calls:
            name = call.function.name
            tool = REGISTRY.get(name)
            if tool is None:
                messages.append(
                    {"role": "tool", "tool_call_id": call.id,
                     "content": json.dumps({"error": "unknown tool"})}
                )
                continue

            payload = _arguments(call)

            if tool.needs_confirmation() and not await approve(tool.name, tool.risk, payload):
                yield AgentEvent("denied", {"tool": tool.name})
                messages.append(
                    {"role": "tool", "tool_call_id": call.id,
                     "content": json.dumps({"error": "denied by operator"})}
                )
                continue

            yield AgentEvent("state", {"state": "tool_execution"})
            yield AgentEvent("tool", {"tool": tool.name, "risk": tool.risk})
            try:
                output = await tool.run(payload)
            except Exception as err:
                log.exception("tool %s failed", tool.name)
                output = {"error": type(err).__name__}

            result.evidence.append({"tool": tool.name, "output": output})

            long_term.mark_tool_used(tool.name)

            if tool.name == "remember" and "remembered" in output:
                yield AgentEvent(
                    "memory",
                    {"id": output["id"], "fact": output["remembered"], "provenance": output["provenance"]},
                )

            if tool.preview and "error" not in output:
                try:
                    yield AgentEvent("preview", tool.preview(output))
                except Exception:
                    log.exception("preview for %s failed", tool.name)

            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(output)}
            )

        yield AgentEvent("state", {"state": "processing"})

    log.warning("hit MAX_TURNS without a final answer")
    result.text = "I wasn't able to finish that within the step budget."


async def emit_memory_event(output: dict) -> AsyncIterator[AgentEvent]:
    """Một kết quả `remember` thành một AgentEvent. Tách ra để test được mà
    không phải dựng cả vòng agent."""
    if "remembered" in output:
        yield AgentEvent(
            "memory",
            {"id": output["id"], "fact": output["remembered"], "provenance": output["provenance"]},
        )
