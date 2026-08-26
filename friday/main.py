"""§2 API layer — one SSE endpoint driving the frontend state machine.

The event names map 1:1 onto the zustand store: `state` -> transition(),
`viz` -> setVisualization(), `answer` -> setAnswer(). Adding a step to the
agent flow means emitting another event, not changing the transport.
"""

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from openai import APIError, NotFoundError

from . import agent, llm
from .planner import plan

# uvicorn configures only its own loggers, so without this the diagnostics
# below never reach Render's log stream — the one place you can read them.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(levelname)s %(name)s: %(message)s",
)

log = logging.getLogger("friday")

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("FRIDAY_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Announce the effective config once, without secrets.

    A misconfigured deploy fails in ways that point somewhere else: a browser
    CORS rejection shows only in the browser console while the server logs a
    clean 200, and a missing key surfaces as an `error` event the UI reports as
    the planner being down. Both are one log line to diagnose from here.
    """
    log.info("planner configured: %s", llm.configured())
    log.info("model: %s at %s", llm.model(), llm.base_url())
    log.info("allowed origins: %s", ALLOWED_ORIGINS)
    if not llm.configured():
        log.warning("no provider key set - every query will return an error event")
    if not ALLOWED_ORIGINS:
        log.warning("FRIDAY_ALLOWED_ORIGINS is empty - the browser will block every origin")
    elif ALLOWED_ORIGINS == ["http://localhost:3000"]:
        log.warning(
            "FRIDAY_ALLOWED_ORIGINS is still the localhost default; "
            "a deployed frontend will be blocked by the browser"
        )
    yield


app = FastAPI(title="FRIDAY Orchestrator", lifespan=lifespan)

# The UI is served from a different origin. Kept to an explicit list — a
# wildcard here would let any page in any tab drive the agent.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["content-type"],
)

#: §11 — in-flight approval requests, keyed by a per-request id.
#: Process-local by design: a pending decision is meaningless after a restart,
#: so it must expire rather than be resumed from a shared store.
PENDING: dict[str, asyncio.Future[bool]] = {}

#: An unanswered prompt must not pin a request open forever.
CONFIRM_TIMEOUT_S = 120


class Query(BaseModel):
    query: str


class Decision(BaseModel):
    id: str
    approved: bool


def sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


async def run_query(query: str) -> AsyncIterator[str]:
    """The §9 flow: think -> (tools, gated) -> plan the hologram -> speak."""
    yield sse("state", {"state": "thinking"})

    outcome = agent.AgentResult(text="")

    # The agent runs as its own task feeding a queue, rather than being iterated
    # directly. It has to be able to block mid-iteration waiting for an operator
    # decision, and a directly-iterated generator cannot yield the approval
    # prompt while it is itself blocked on the answer to that prompt — the
    # request would deadlock until the approval timed out.
    events: asyncio.Queue[agent.AgentEvent | None] = asyncio.Queue()

    async def approve(tool: str, risk: str, payload: dict[str, Any]) -> bool:
        request_id = uuid.uuid4().hex
        decided: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        PENDING[request_id] = decided
        await events.put(
            agent.AgentEvent("confirm", {"id": request_id, "tool": tool, "risk": risk, "input": payload})
        )
        try:
            return await asyncio.wait_for(decided, CONFIRM_TIMEOUT_S)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            # Silence is not consent.
            log.info("approval for %s timed out", tool)
            return False
        finally:
            PENDING.pop(request_id, None)

    failure: BaseException | None = None

    async def pump() -> None:
        nonlocal failure
        try:
            async for event in agent.run(query, approve, outcome):
                await events.put(event)
        except BaseException as err:  # surfaced to the caller below
            failure = err
        finally:
            await events.put(None)

    task = asyncio.create_task(pump())
    # Which half failed. "planner unavailable" on an agent-loop error sends the
    # operator to the wrong file.
    stage = "agent"
    try:
        while True:
            event = await events.get()
            if event is None:
                break

            # §18 — a preview is a real visualization, just an early one. It
            # goes out on the same `viz` event so the UI needs no new concept;
            # the planner's spec later replaces it.
            #
            # No `visualizing` state here on purpose. FRIDAY is still running
            # tools, and the UI's transition table has no visualizing ->
            # tool_execution edge: announcing it would strand the HUD on
            # VISUALIZING for the rest of the turn, since guarded transitions
            # drop illegal edges silently.
            if event.kind == "preview":
                yield sse("viz", {"animation": "materialize", "interaction": "none", **event.payload})
                continue

            yield sse(event.kind, event.payload)

        if failure is not None:
            raise failure

        stage = "planner"
        result = await plan(query, outcome.text, outcome.evidence)
    except NotFoundError:
        # Almost always a stale or misspelled model name — say which one, or
        # the operator is left guessing at an opaque 404.
        log.exception("model %r not available at %s", llm.model(), llm.base_url())
        yield sse("error", {"message": f"model '{llm.model()}' unavailable at this endpoint"})
        yield sse("state", {"state": "error"})
        yield sse("done", {})
        return
    except APIError as err:
        log.exception("model call failed")
        yield sse("error", {"message": f"{stage} error: {type(err).__name__}"})
        yield sse("state", {"state": "error"})
        yield sse("done", {})
        return
    except Exception:
        log.exception("query failed in %s stage", stage)
        yield sse("error", {"message": f"{stage} unavailable"})
        yield sse("state", {"state": "error"})
        yield sse("done", {})
        return
    finally:
        # A client that hangs up mid-approval must not leave the agent running.
        if not task.done():
            task.cancel()

    yield sse("state", {"state": "visualizing"})
    yield sse("viz", result.model_dump(exclude={"answer"}, exclude_none=True))
    yield sse("state", {"state": "speaking"})
    # The agent's own words win — the planner only restates them.
    yield sse("answer", {"text": outcome.text or result.answer})
    yield sse("done", {})


@app.post("/query")
async def query_endpoint(body: Query) -> StreamingResponse:
    return StreamingResponse(
        run_query(body.query),
        media_type="text/event-stream",
        # nginx buffers SSE into uselessness without this
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/confirm")
async def confirm_endpoint(body: Decision) -> dict[str, Any]:
    """§11 — the operator's answer to a high-risk tool call."""
    decided = PENDING.get(body.id)
    if decided is None or decided.done():
        # Already answered, expired, or never existed. All three are "no longer
        # actionable" from the caller's side.
        raise HTTPException(status_code=404, detail="no pending decision with that id")
    decided.set_result(body.approved)
    return {"ok": True, "approved": body.approved}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "planner": llm.configured(),
        "model": llm.model(),
        "endpoint": llm.base_url(),
    }
