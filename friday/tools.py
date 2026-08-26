"""§10 Tool system and §11 risk tiers.

Each tool declares a risk level; the orchestrator decides from that whether the
call needs operator approval. The tool itself never asks — that decision does
not belong next to the implementation.

§22 is explicit that the model must not reach a shell. There is deliberately no
shell, no `eval`, and no arbitrary-path write here: `write_note` is confined to
a single directory by construction, not by asking the model nicely.
"""

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import psutil

RiskLevel = Literal["low", "medium", "high"]

#: §18 — turns a tool's raw output into a partial visualization spec so the
#: hologram can materialize the moment data lands, instead of waiting for the
#: whole turn to finish. Deterministic on purpose: a model call per tool result
#: would triple our request count against a rate-limited free tier.
Preview = Callable[[dict[str, Any]], dict[str, Any]]

#: Anything above this needs a human to say yes (§11).
CONFIRM_ABOVE: set[RiskLevel] = {"high"}

NOTES_DIR = Path(__file__).resolve().parent.parent / "notes"


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    risk: RiskLevel
    run: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
    #: None means this tool has nothing worth showing before the planner runs.
    preview: Preview | None = None

    def needs_confirmation(self) -> bool:
        return self.risk in CONFIRM_ABOVE

    def as_api_tool(self) -> dict[str, Any]:
        """The OpenAI function-calling shape. Risk stays server-side —
        the model has no say in its own permissions."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


async def _system_metrics(_: dict[str, Any]) -> dict[str, Any]:
    """Real numbers from this host, so the gauges are measurements."""
    # interval=None returns the value since the last call, non-blocking
    psutil.cpu_percent(interval=None)
    disk = psutil.disk_usage(str(Path.home().anchor or "/"))
    return {
        "cpu_percent": round(psutil.cpu_percent(interval=0.15), 1),
        "memory_percent": round(psutil.virtual_memory().percent, 1),
        "disk_percent": round(disk.percent, 1),
        "cpu_count": psutil.cpu_count(logical=True),
    }


def _metrics_preview(output: dict[str, Any]) -> dict[str, Any]:
    """Three host readings are gauge-shaped; no model needed to know that."""
    return {
        "type": "radial_gauge",
        "title": "SYSTEM LOAD",
        "data": {
            "metrics": [
                {"label": "CPU", "value": output.get("cpu_percent", 0), "unit": "%"},
                {"label": "RAM", "value": output.get("memory_percent", 0), "unit": "%"},
                {"label": "DISK", "value": output.get("disk_percent", 0), "unit": "%"},
            ]
        },
    }


async def _process_list(payload: dict[str, Any]) -> dict[str, Any]:
    """Top processes by memory share.

    Ranked by memory, not CPU: `cpu_percent` reads 0.0 the first time it is
    sampled for a process, so a CPU ranking here would be noise wearing a
    number.
    """
    try:
        limit = max(1, min(10, int(payload.get("limit", 5))))
    except (TypeError, ValueError):
        limit = 5

    seen: list[tuple[str, float]] = []
    for proc in psutil.process_iter(["name", "memory_percent"]):
        try:
            info = proc.info
            if info["name"] and info["memory_percent"]:
                seen.append((info["name"], round(info["memory_percent"], 2)))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue  # processes come and go mid-scan; that is normal

    seen.sort(key=lambda row: row[1], reverse=True)
    top = seen[:limit]
    return {"ranked_by": "memory_percent", "processes": [{"name": n, "percent": v} for n, v in top]}


def _process_preview(output: dict[str, Any]) -> dict[str, Any]:
    rows = output.get("processes", [])
    return {
        "type": "bar_3d",
        "title": "TOP PROCESSES",
        "data": {"series": [{"label": "MEM", "points": [row["percent"] for row in rows]}]},
    }


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]")


async def _write_note(payload: dict[str, Any]) -> dict[str, Any]:
    """Write a note. High risk: it is the only thing here that persists."""
    raw = str(payload.get("name", "")).strip()
    body = str(payload.get("body", ""))

    # Sanitise to a bare stem, then rebuild the path ourselves. Nothing the
    # model sends can escape NOTES_DIR because nothing it sends is used as a
    # path component verbatim.
    stem = _SAFE_NAME.sub("-", raw)[:64].strip("-")
    if not stem:
        return {"error": "invalid note name"}

    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    target = NOTES_DIR / f"{stem}.md"
    target.write_text(body, encoding="utf-8")
    return {"written": target.name, "bytes": len(body.encode("utf-8"))}


REGISTRY: dict[str, Tool] = {
    tool.name: tool
    for tool in [
        Tool(
            name="get_system_metrics",
            description=(
                "Read live CPU, memory and disk utilisation from the host this "
                "orchestrator runs on. Returns percentages 0-100."
            ),
            input_schema={"type": "object", "properties": {}, "required": []},
            risk="low",
            run=_system_metrics,
            preview=_metrics_preview,
        ),
        Tool(
            name="get_process_list",
            description=(
                "List the processes using the most memory on this host, ranked. "
                "Use after get_system_metrics when memory looks high."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "how many to return, 1-10"}
                },
                "required": [],
            },
            risk="low",
            run=_process_list,
            preview=_process_preview,
        ),
        Tool(
            name="write_note",
            description="Persist a short markdown note to the operator's notes directory.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "short slug, no extension"},
                    "body": {"type": "string", "description": "markdown body"},
                },
                "required": ["name", "body"],
            },
            risk="high",
            run=_write_note,
        ),
    ]
}


def api_tools() -> list[dict[str, Any]]:
    return [tool.as_api_tool() for tool in REGISTRY.values()]
