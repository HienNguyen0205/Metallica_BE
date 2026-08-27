"""API request/response schemas."""

from pydantic import BaseModel


class Query(BaseModel):
    query: str


class Decision(BaseModel):
    id: str
    approved: bool
