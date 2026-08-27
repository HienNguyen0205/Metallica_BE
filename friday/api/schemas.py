"""API request/response schemas."""

from pydantic import BaseModel, Field


class Query(BaseModel):
    query: str
    #: §15 — opaque, client-generated, and used only as a dict key. Bounded
    #: because it arrives from a public endpoint and an unbounded string would
    #: be stored verbatim.
    session_id: str | None = Field(default=None, max_length=64)


class Decision(BaseModel):
    id: str
    approved: bool
