import json
import logging
from typing import Any

from openai import BadRequestError

from friday import llm
from friday.schemas.visualization import VisualizationPlan

from .prompts import SYSTEM

log = logging.getLogger("friday.planner")


def _schema() -> dict[str, Any]:
    return VisualizationPlan.model_json_schema()


async def plan(query: str, answer: str, evidence: list[dict[str, Any]]) -> VisualizationPlan:
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
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()
