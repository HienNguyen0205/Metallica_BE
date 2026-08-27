"""Tool registry — register / get / list / resolve."""

from typing import Any

from .base import Tool
from .filesystem.notes import run_write_note
from .system.metrics import preview_metrics, run_system_metrics
from .system.processes import preview_processes, run_process_list


def _build_default_registry() -> dict[str, Tool]:
    tools = [
        Tool(
            name="get_system_metrics",
            description=(
                "Read live CPU, memory and disk utilisation from the host this "
                "orchestrator runs on. Returns percentages 0-100."
            ),
            input_schema={"type": "object", "properties": {}, "required": []},
            risk="low",
            run=run_system_metrics,
            preview=preview_metrics,
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
            run=run_process_list,
            preview=preview_processes,
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
            run=run_write_note,
        ),
    ]
    return {t.name: t for t in tools}


REGISTRY: dict[str, Tool] = _build_default_registry()

# Keep legacy alias for tests that import NOTES_DIR from tools
from .filesystem.notes import NOTES_DIR  # noqa: E402

# Re-export for shortcuts
from .base import CONFIRM_ABOVE  # noqa: E402


def register(tool: Tool) -> None:
    REGISTRY[tool.name] = tool


def get(name: str) -> Tool | None:
    return REGISTRY.get(name)


def list_tools() -> list[Tool]:
    return list(REGISTRY.values())


def api_tools() -> list[dict[str, Any]]:
    return [tool.as_api_tool() for tool in REGISTRY.values()]
