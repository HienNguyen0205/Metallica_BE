"""API routes — extracted from main.py."""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from openai import APIError, NotFoundError, RateLimitError

from friday import agent, llm, memory
from friday.api import dependencies as deps
from friday.api.dependencies import PENDING, guard, require_known_origin
from friday.api.schemas import Decision, Query
from friday.core.config import settings
from friday.events.serializer import sse
from friday.memory import consolidate
from friday.memory import embed as embed_mod
from friday.memory import long_term
from friday.memory.embed import EmbedError

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

#: Strong references for fire-and-forget background tasks (consolidation).
#: asyncio only holds a *weak* ref to a task, so one nothing else points to
#: can be GC'd before it ever runs — dropped silently, since `consolidate.run`
#: swallows its own exceptions. Each task removes itself on completion, so
#: this stays bounded.
_BACKGROUND_TASKS: set[asyncio.Task] = set()


def quota_detail(err: Exception) -> str:
    """Which limit the provider refused on, and when it clears.

    This used to read "the API key is over quota" for every 429, which was
    wrong in the common case and expensively so: the free tier's binding limit
    is requests *per minute*, and one query spends two or three of them, so a
    normal run trips it at around the sixth question. Someone reading "over
    quota" goes looking at billing for a wall that clears in twelve seconds.

    Every field is optional. Only Gemini's OpenAI shim is known to send this
    shape, and it wraps the object in a list; anything else falls through to
    the bare message rather than guessing.
    """
    body = getattr(err, "body", None)
    if isinstance(body, list):
        body = body[0] if body else None
    if not isinstance(body, dict):
        return ""

    quota_id = ""
    retry = ""
    for entry in (body.get("error") or {}).get("details") or []:
        if not isinstance(entry, dict):
            continue
        for violation in entry.get("violations") or []:
            quota_id = violation.get("quotaId") or quota_id
        retry = entry.get("retryDelay") or retry

    if "PerMinute" in quota_id:
        window = " - requests-per-minute limit, not the daily quota"
    elif "PerDay" in quota_id:
        window = " - the daily quota for this model is spent"
    else:
        window = ""

    return f"{window}{f'; retry in {retry}' if retry else ''}"


async def recall_block(query: str) -> str:
    """Ký ức liên quan tới câu hỏi này, đã đóng gói cho prompt.

    Chuỗi rỗng là câu trả lời hợp lệ và là mặc định khi có bất cứ gì hỏng: không
    có ký ức, không cấu hình store, embedding chết. Ký ức là phần thêm — không
    lý do gì để một câu hỏi thất bại vì FRIDAY không nhớ ra.
    """
    if not long_term.CACHE:
        return ""
    try:
        vectors = await embed_mod.embed([query])
    except EmbedError:
        log.warning("recall skipped: embedding unavailable", exc_info=True)
        return ""

    hits = long_term.top_k(vectors[0], long_term.TOP_K_DEFAULT)
    if hits:
        asyncio.get_running_loop().run_in_executor(None, _touch, [m.id for m in hits])
    return long_term.render_block(hits)


def _touch(ids: list[int]) -> None:
    from friday.memory.store import touch

    touch(ids)


async def run_query(query: str, session_id: str | None = None) -> AsyncIterator[str]:
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
            async for event in agent.run(query, approve, outcome, memory.history(session_id), memories):
                await events.put(event)
        except BaseException as err:
            failure = err
        finally:
            await events.put(None)

    memories = await recall_block(query)
    task = asyncio.create_task(pump())
    stage = "agent"
    # §18 — the component the last preview already materialised. The planner is
    # pinned to it rather than allowed to re-decide: a preview is derived from
    # the tool's own output shape without a model call, so it is both right and
    # the same every run, while the planner re-reads the same JSON and reaches a
    # different conclusion often enough to be visible — the user watches bars
    # build and then get replaced by gauges for no reason they can see.
    pinned_type: str | None = None
    try:
        while True:
            event = await events.get()
            if event is None:
                break

            if event.kind == "preview":
                pinned_type = event.payload.get("type") or pinned_type
                yield sse("viz", {"animation": "materialize", "interaction": "none", **event.payload})
                continue

            yield sse(event.kind, event.payload)

        if failure is not None:
            raise failure

        stage = "planner"
        plan_fn = await _get_plan()
        result = await plan_fn(query, outcome.text, outcome.evidence, pinned_type)
    except NotFoundError:
        log.exception("model %r not available at %s", llm.model(), llm.base_url())
        yield sse("error", {"message": f"model '{llm.model()}' unavailable at this endpoint"})
        yield sse("state", {"state": "error"})
        yield sse("done", {})
        return
    except RateLimitError as err:
        log.exception("provider rate limit hit for model %r", llm.model())
        yield sse(
            "error",
            {"message": f"provider rate limit reached ({stage}){quota_detail(err)}"},
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
    answer = outcome.text or result.answer
    # Recorded only once the turn has actually produced an answer — a failed
    # turn returns above, so a provider outage cannot poison the session with
    # an exchange that never happened.
    memory.remember(session_id, query, answer)
    yield sse("answer", {"text": answer})
    yield sse("done", {})
    # Sau `done`, không chờ: nó tốn một model call và người dùng không có lý do
    # gì phải đợi FRIDAY dọn dẹp.
    consolidate.note_turn()
    if consolidate.should_run():
        bg_task = asyncio.create_task(consolidate.run())
        _BACKGROUND_TASKS.add(bg_task)
        bg_task.add_done_callback(_BACKGROUND_TASKS.discard)


@router.post("/query", dependencies=[Depends(guard)])
async def query_endpoint(body: Query) -> StreamingResponse:
    return StreamingResponse(
        run_query(body.query, body.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/confirm", dependencies=[Depends(require_known_origin)])
async def confirm_endpoint(body: Decision) -> dict[str, Any]:
    decided = PENDING.get(body.id)
    if decided is None or decided.done():
        raise HTTPException(status_code=404, detail="no pending decision with that id")
    decided.set_result(body.approved)
    return {"ok": True, "approved": body.approved}


@router.get("/memory", dependencies=[Depends(require_known_origin)])
async def list_memory() -> dict[str, Any]:
    """Mọi thứ FRIDAY nhớ. Không tính vào rate limit — không có model call nào.

    Vector không nằm trong response: 768 số float không nói gì với người đọc và
    làm payload phình lên vô ích.
    """
    return {
        "memories": [
            {
                "id": m.id,
                "fact": m.fact,
                "provenance": m.provenance,
                "use_count": m.use_count,
                "last_used_at": m.last_used_at,
            }
            for m in long_term.CACHE
        ]
    }


@router.delete("/memory/{memory_id}", dependencies=[Depends(require_known_origin)])
async def forget_memory(memory_id: int) -> dict[str, Any]:
    return {"ok": long_term.forget(memory_id)}


@router.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "planner": llm.configured(),
        "model": llm.model(),
        "endpoint": llm.base_url(),
        # Reported so a deploy's limits can be read off the running service
        # rather than inferred from which env vars someone remembered to set.
        "limits": {
            "per_client_hourly": settings.rate_limit_per_hour,
            "global_hourly": settings.global_limit_per_hour,
        },
    }
