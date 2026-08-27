"""Agent context — placeholder for Memory/RAG integration (§15-17)."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentContext:
    user_query: str
    conversation: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    memory: list[dict[str, Any]] | None = None
    retrieved_documents: list[dict[str, Any]] | None = None
    session: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
