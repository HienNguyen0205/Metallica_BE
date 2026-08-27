"""Shared API dependencies / state."""

import asyncio

# §11 — in-flight approval requests, keyed by per-request id.
PENDING: dict[str, asyncio.Future[bool]] = {}

CONFIRM_TIMEOUT_S = 120
