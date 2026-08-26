"""§9 agent loop — plan, call tools, gather results.

A manual loop rather than an SDK helper: it has to yield a state change before
each tool runs and pause mid-iteration waiting for an operator decision, which
a batteries-included runner cannot interleave into an SSE stream.
"""

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from . import llm
from .tools import REGISTRY, api_tools

log = logging.getLogger("friday.agent")

#: Stops a pathological plan from looping on tools forever.
MAX_TURNS = 6

SYSTEM = """You are FRIDAY, an AI operations interface.

Use your tools to answer from real data rather than estimating. \
`get_system_metrics` reads the machine this orchestrator runs on — call it \
whenever the question touches system load, CPU, memory or disk. Do not answer \
those questions from memory or guesswork; you have no numbers until a tool \
gives you some.

If a tool is denied by the operator, say so plainly; do not retry it and do not \
substitute invented figures for the data you were refused.

When you have what you need, answer in one or two calm, factual sentences."""

#: Called with (tool_name, risk, tool_input); returns True to allow the call.
Approver = Callable[[str, str, dict[str, Any]], Awaitable[bool]]


@dataclass
class AgentEvent:
    """Something the transport should tell the UI about."""

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    text: str
    evidence: list[dict[str, Any]] = field(default_factory=list)


def _echo(call: Any) -> dict[str, Any]:
    """Replay a tool call for the next turn, keeping provider-specific extras.

    Rebuilding this by hand from id/name/arguments loses fields the provider
    requires back. Gemini attaches a `thought_signature` under
    `extra_content.google` and rejects the following turn with a 400 if it is
    missing. Round-tripping every non-null field keeps that working without
    hard-coding one vendor's quirk.
    """
    return {key: value for key, value in call.model_dump().items() if value is not None}


def _arguments(call: Any) -> dict[str, Any]:
    """Tool arguments arrive as a model-generated JSON string — parse, never
    string-match, and never trust that it parses at all."""
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
) -> AsyncIterator[AgentEvent]:
    """Drive the tool loop, yielding progress. Fills `result` in place."""
    api = llm.client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
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
            except Exception as err:  # a broken tool must not kill the turn
                log.exception("tool %s failed", tool.name)
                output = {"error": type(err).__name__}

            result.evidence.append({"tool": tool.name, "output": output})

            # §18 — show it the moment it lands rather than at the end of the
            # turn. Deterministic, so this costs no extra model call.
            if tool.preview and "error" not in output:
                try:
                    yield AgentEvent("preview", tool.preview(output))
                except Exception:
                    # A malformed preview must never cost us the tool result.
                    log.exception("preview for %s failed", tool.name)

            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(output)}
            )

        yield AgentEvent("state", {"state": "processing"})

    log.warning("hit MAX_TURNS without a final answer")
    result.text = "I wasn't able to finish that within the step budget."
