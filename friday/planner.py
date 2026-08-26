"""§5 Visualization Planner — the LLM half.

The rules planner in src/lib/vizPlanner.ts stays as the fallback: when the API
key is missing or the call fails, the UI still answers rather than dying. The
doc calls for `Rules + LLM + heuristics`; this is the LLM leg.

Kept separate from the agent loop on purpose. The agent decides what is true;
this decides how to show it. Feeding it the tool output as evidence is what
keeps the gauges reporting measurements instead of plausible-looking numbers.
"""

import json
import logging
from typing import Any

from openai import BadRequestError

from . import llm
from .schema import VisualizationPlan

log = logging.getLogger("friday.planner")

SYSTEM = """You choose how FRIDAY displays an answer.

You do not draw. You pick one visualization component from a fixed set and \
supply the data it renders:

- radial_gauge — one or more percentages (CPU, memory, disk usage)
- health_core — a single overall status or score
- radar — scanning, threat or contact detection
- waveform — audio, signal or frequency
- line_3d — a trend or time series (use `series`)
- bar_3d — comparing discrete magnitudes (use `series`, one entry)
- timeline — an ordered sequence of events (use `events`, `at` from 0 to 1)
- network — topology, dependencies, service graphs (use `nodes` and `links`)
- globe — geography, regions, edge locations (use `points`)
- particle_flow — traffic, throughput, streaming volume

Rules:
- Fill only the data fields the chosen type reads. Omit the rest.
- Metric values are 0-100.
- Titles are short and all-caps. Labels are short and all-caps.
- Use the measured evidence verbatim where it is given. Never round a measured \
value into a rounder-looking one, and never add data points that were not \
measured to make a chart look fuller.
- If there is no evidence, choose a component that shows the shape of the \
answer without implying precision you do not have.
- `answer` is what FRIDAY says aloud: one or two sentences, calm and factual.

Reply with JSON only — no prose, no markdown fence."""


def _schema() -> dict[str, Any]:
    return VisualizationPlan.model_json_schema()


async def plan(query: str, answer: str, evidence: list[dict[str, Any]]) -> VisualizationPlan:
    """Turn an answered question into a spec. Raises on API failure."""
    context = f"Question: {query}\n\nFRIDAY's answer: {answer}"
    if evidence:
        context += f"\n\nMeasured evidence:\n{json.dumps(evidence, indent=2)}"
    else:
        context += "\n\nMeasured evidence: none — no tool was run."

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": context},
    ]

    text = await _complete(messages)
    return VisualizationPlan.model_validate_json(_strip_fence(text))


async def _complete(messages: list[dict[str, Any]]) -> str:
    """Ask for schema-constrained JSON, degrading if the provider can't do it.

    Support for `json_schema` varies across OpenAI-compatible providers — some
    reject Pydantic's `$defs`/`$ref` output outright. Rather than pin this file
    to one vendor's quirks, fall back to plain JSON mode with the schema in the
    prompt, which every provider in this class supports.
    """
    api = llm.client()
    try:
        response = await api.chat.completions.create(
            model=llm.model(),
            messages=messages,  # type: ignore[arg-type]
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "visualization_plan", "schema": _schema()},
            },
        )
        return response.choices[0].message.content or ""
    except BadRequestError as err:
        log.warning("provider rejected json_schema, falling back to json_object: %s", err)

    relaxed = [
        *messages,
        {"role": "system", "content": f"Match this JSON Schema exactly:\n{json.dumps(_schema())}"},
    ]
    response = await api.chat.completions.create(
        model=llm.model(),
        messages=relaxed,  # type: ignore[arg-type]
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def _strip_fence(text: str) -> str:
    """Some models wrap JSON in a markdown fence despite being told not to."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()
