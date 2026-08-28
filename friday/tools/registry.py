"""Tool registry — register / get / list / resolve."""

from typing import Any

from .base import Tool
from .filesystem.notes import run_write_note
from .integrations.search import run_search_web
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
            name="search_web",
            description=(
                "Search the public web and return extracts from the top results. "
                "Use for anything this host cannot measure itself: current events, "
                "documentation, prices, weather, or any fact you are unsure of. "
                "Prefer this over answering from memory when the answer could have "
                "changed since training. Results come from a search index, not "
                "live sources: for fast-moving numbers such as prices, scores or "
                "weather, say the value is approximate and may lag rather than "
                "presenting it as the current one."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "the search query, in the language of the source you expect",
                    }
                },
                "required": ["query"],
            },
            # §11 LOW, matching the doc's own example list. It reads public data
            # and writes nothing — but note it is the only tool that puts text
            # from strangers into the model's context. That is a prompt-injection
            # surface, and the containment is that every consequential tool is
            # risk="high" and therefore blocked on a human, so a page telling
            # FRIDAY to write a note still has to get past the operator.
            risk="low",
            run=run_search_web,
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
