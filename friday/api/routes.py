"""API routes — extracted from main.py."""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai import APIError, NotFoundError, RateLimitError

from friday import agent, llm
from friday.api import dependencies as deps
from friday.api.dependencies import PENDING
from friday.api.schemas import Decision, Query
from friday.events.serializer import sse

# Keep CONFIRM_TIMEOUT_S readable for old imports, but resolve dynamically
CONFIRM_TIMEOUT_S = deps.CONFIRM_TIMEOUT_S


def _get_confirm_timeout() -> float:
    # Tests monkey-patch friday.main.CONFIRM_TIMEOUT_S; respect that live value.
    try:
        import friday.main as main_mod

        val = getattr(main_mod, "CONFIRM_TIMEOUT_S", None)
        if isinstance(val, (int, float)):
            return float(val)
    except ImportError:
        pass
    return float(deps.CONFIRM_TIMEOUT_S)


async def _get_plan():
    # Tests monkey-patch friday.main.plan; respect that if set.
    try:
        import friday.main as main_mod

        maybe = getattr(main_mod, "plan", None)
        # If main.plan was overridden to a fake, use it (check if it's not the original import)
        if maybe is not None:
            import friday.planner as planner_mod

            if maybe is not planner_mod.plan:
                return maybe
    except ImportError:
        pass
    from friday.planner import plan as real_plan

    return real_plan

log = logging.getLogger("friday")

router = APIRouter()


async def run_query(query: str) -> AsyncIterator[str]:
    yield sse("state", {"state": "thinking"})

    outcome = agent.AgentResult(text="")
    events: asyncio.Queue[agent.AgentEvent | None] = asyncio.Queue()

    async def approve(tool: str, risk: str, payload: dict[str, Any]) -> bool:
        request_id = uuid.uuid4().hex
        decided: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        PENDING[request_id] = decided
        await events.put(
            agent.AgentEvent("confirm", {"id": request_id, "tool": tool, "risk": risk, "input": payload})
        )
        try:
            return await asyncio.wait_for(decided, _get_confirm_timeout())
        except (asyncio.TimeoutError, asyncio.CancelledError):
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
        except BaseException as err:
            failure = err
        finally:
            await events.put(None)

    task = asyncio.create_task(pump())
    stage = "agent"
    try:
        while True:
            event = await events.get()
            if event is None:
                break

            if event.kind == "preview":
                yield sse("viz", {"animation": "materialize", "interaction": "none", **event.payload})
                continue

            yield sse(event.kind, event.payload)

        if failure is not None:
            raise failure

        stage = "planner"
        plan_fn = await _get_plan()
        result = await plan_fn(query, outcome.text, outcome.evidence)
    except NotFoundError:
        log.exception("model %r not available at %s", llm.model(), llm.base_url())
        yield sse("error", {"message": f"model '{llm.model()}' unavailable at this endpoint"})
        yield sse("state", {"state": "error"})
        yield sse("done", {})
        return
    except RateLimitError:
        log.exception("provider rate limit hit for model %r", llm.model())
        yield sse(
            "error",
            {"message": f"provider rate limit reached ({stage}) - the API key is over quota"},
        )
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
        if not task.done():
            task.cancel()

    yield sse("state", {"state": "visualizing"})
    yield sse("viz", result.model_dump(exclude={"answer"}, exclude_none=True))
    yield sse("state", {"state": "speaking"})
    yield sse("answer", {"text": outcome.text or result.answer})
    yield sse("done", {})


@router.post("/query")
async def query_endpoint(body: Query) -> StreamingResponse:
    return StreamingResponse(
        run_query(body.query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/confirm")
async def confirm_endpoint(body: Decision) -> dict[str, Any]:
    decided = PENDING.get(body.id)
    if decided is None or decided.done():
        raise HTTPException(status_code=404, detail="no pending decision with that id")
    decided.set_result(body.approved)
    return {"ok": True, "approved": body.approved}


@router.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "planner": llm.configured(),
        "model": llm.model(),
        "endpoint": llm.base_url(),
    }
