"""Application lifespan — startup diagnostics without secrets."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from friday.llm import base_url, configured, model
from .config import settings

log = logging.getLogger("friday")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    log.info("planner configured: %s", configured())
    log.info("model: %s at %s", model(), base_url())
    log.info("allowed origins: %s", settings.allowed_origins)
    if not configured():
        log.warning("no provider key set - every query will return an error event")
    if not settings.allowed_origins:
        log.warning("FRIDAY_ALLOWED_ORIGINS is empty - the browser will block every origin")
    elif settings.allowed_origins == ["http://localhost:3000"]:
        log.warning(
            "FRIDAY_ALLOWED_ORIGINS is still the localhost default; "
            "a deployed frontend will be blocked by the browser"
        )
    yield
