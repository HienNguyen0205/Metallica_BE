# FRIDAY Orchestrator

The backend half of §2. One SSE endpoint drives the frontend state machine.

> Paths like `src/lib/store.ts` below refer to the **Metallica UI repository**,
> which is separate from this one. The two are contract-coupled but deploy
> independently: this service owns the event contract, the UI consumes it.

## Run

```bash
cd backend
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS / Linux

# free key, no card: https://aistudio.google.com/apikey
export GEMINI_API_KEY=...
./.venv/Scripts/python.exe -m uvicorn friday.main:app --port 8000 --reload
```

The frontend points at `http://localhost:8000` by default; override with
`NEXT_PUBLIC_FRIDAY_API` in `.env.local`.

`GET /health` reports what the gateway is pointed at:

```json
{ "ok": true, "planner": true, "model": "gemini-2.5-flash", "endpoint": "..." }
```

## §8 Model gateway

Every model call goes through an **OpenAI-compatible** endpoint (`friday/llm.py`),
so the provider is configuration rather than code. Defaults to Gemini's free
tier; three env vars move it anywhere else:

| Provider  | `FRIDAY_LLM_BASE_URL`                                       | Notes                       |
| --------- | ----------------------------------------------------------- | --------------------------- |
| Gemini    | *(default)*                                                  | ~1500 req/day free, no card |
| Groq      | `https://api.groq.com/openai/v1`                             | fastest; smaller models     |
| Cerebras  | `https://api.cerebras.ai/v1`                                 | high daily tokens, low RPM  |
| OpenRouter| `https://openrouter.ai/api/v1`                               | aggregator, some free models|
| Ollama    | `http://localhost:11434/v1`                                  | local, offline, no key      |

Budget roughly **2-3 model calls per user query** (one agent turn, one more
after a tool returns, one planner call). Requests-per-minute limits bite here
long before token limits do.

Free tiers usually permit training on your prompts — fine for a demo, not for
real data.

## The event contract

`POST /query` with `{"query": "..."}` returns `text/event-stream`. Each event
maps onto exactly one store action in `src/lib/store.ts`:

| Event     | Payload                                 | Frontend effect          |
| --------- | --------------------------------------- | ------------------------ |
| `state`   | `{"state": "thinking"}`                 | `transition(state)`      |
| `tool`    | `{"tool": "...", "risk": "low"}`        | (progress only)          |
| `confirm` | `{"id", "tool", "risk", "input"}`       | `setPendingConfirm()`    |
| `denied`  | `{"tool": "..."}`                       | (progress only)          |
| `viz`     | a `VisualizationSpec`                   | `setVisualization()`     |
| `answer`  | `{"text": "..."}`                       | `setAnswer(text)`        |
| `error`   | `{"message": "..."}`                    | logged, flow continues   |
| `done`    | `{}`                                    | stream complete          |

Adding a step to the agent flow means emitting another event — the transport
does not change.

## §18 Streaming visualization

The hologram materializes as results land, not after the turn ends. A tool can
declare a `preview` (`friday/tools.py`) that maps its raw output to a partial
spec; each one goes out as a `viz` event the moment the tool returns, and the
planner's spec replaces it at the end. A single query therefore emits several
`viz` events:

```
thinking → tool_execution → viz(radial_gauge) → viz(bar_3d)
         → processing → visualizing → viz(final) → speaking
```

Previews are **deterministic** — no model call. A planner call per tool result
would triple our request count against a tier limited by requests-per-minute.

A preview deliberately does **not** emit a `visualizing` state. FRIDAY is still
running tools, and the UI's transition table has no `visualizing →
tool_execution` edge; announcing it strands the HUD on VISUALIZING for the rest
of the turn, because guarded transitions drop illegal edges silently with no
error anywhere. `test_stream.py` mirrors that table and walks every emitted
sequence against it, and the UI repo asserts the same invariant from the other
side: `PROCESSING` must appear before the first `VISUALIZING`, since a stranded
HUD never shows `TOOL EXECUTION` again and any check against *that* passes in
both the healthy and the broken case.

Note the endpoint is **POST**, so the frontend uses `fetch` + a stream reader
rather than `EventSource`. `EventSource` is GET-only, which would put the
user's question in the URL and therefore in every access log along the way.

## §10 Tools and §11 permissions

Tools are declared in `friday/tools.py` with a risk level. The orchestrator —
not the tool, and never the model — decides from that level whether a call needs
a human. Risk is stripped from the definition sent to the API: the model has no
say in its own permissions.

| Tool                 | Risk | Effect                                    |
| -------------------- | ---- | ----------------------------------------- |
| `get_system_metrics` | low  | reads host CPU / memory / disk (psutil)   |
| `get_process_list`   | low  | top processes by memory share             |
| `write_note`         | high | writes a markdown file under `notes/`     |

`get_process_list` ranks by memory, not CPU: `cpu_percent` reads 0.0 the first
time a process is sampled, so a CPU ranking there would be noise wearing a
number.

A high-risk call emits `confirm` and **blocks**. The stream stays open while the
UI shows the tool name and its exact arguments; `POST /confirm {id, approved}`
releases it. Silence is not consent — after `CONFIRM_TIMEOUT_S` (120s) the call
is refused and the model is told it was denied.

Per §22 there is no shell tool, no `eval`, and no arbitrary-path write.
`write_note` sanitises the model's string to a bare stem and rebuilds the path
itself, so nothing the model sends is ever used as a path component verbatim.

`get_system_metrics` exists mainly so the gauges show measurements. Without a
tool the planner has nothing but the model's prior, and a chart of invented
numbers is indistinguishable from a real one.

### Why the agent runs as a task, not a loop

`run_query` pumps `agent.run` through a queue rather than iterating it directly.
The approval callback blocks *inside* the agent generator, so a directly
iterated generator could not yield the approval prompt while it was itself
blocked waiting for the answer to that prompt — every high-risk call would stall
until it timed out.

## Why the model never draws

Per §25, Claude does not emit rendering code. It returns a `VisualizationPlan`
(`friday/schema.py`) naming one of ten components the frontend already knows,
plus the data that component reads. The schema is enforced by structured
outputs, so a malformed plan is a validation error rather than a broken scene.

Support for `json_schema` varies across OpenAI-compatible providers — some
reject Pydantic's `$defs`/`$ref` output. `planner.py` degrades to plain JSON
mode with the schema in the prompt when the provider returns a 400, and strips
markdown fences from models that add them anyway.

`friday/schema.py` must stay in lockstep with `VisualizationSpec` in
`src/lib/store.ts`. A field here that does not exist there renders as nothing.

## Fallback behaviour

If the orchestrator is unreachable the UI falls back to the local rules planner
in `src/lib/vizPlanner.ts` and logs a warning. That path serves **canned demo
data** — it exists so the interface is presentable with no backend running, not
as a degraded live mode.

## Tests

```bash
./.venv/Scripts/python.exe test_stream.py     # transport + permissions
./.venv/Scripts/python.exe test_provider.py   # gateway against a fake provider
```

`test_stream.py` covers the event sequence, the approval gate (announced before
running, denial reported to the model, timeout refuses), that a dead model still
closes the stream rather than leaving the UI stuck in `THINKING`, and that a
hostile note name cannot escape `notes/`. Model calls are stubbed.

`test_provider.py` runs the **real** agent loop and planner against a local
fake OpenAI-compatible server: the tool-call round trip, evidence reaching the
planner, and the `json_schema` -> `json_object` fallback. No key, no network.

## Not built yet

Memory (§15), RAG (§16), vision (§14) and voice (§12/§13) have no
implementation. Approvals are process-local, so the orchestrator is
single-instance until they move to shared state.

Model choice is measured, not assumed: `gemini-2.5-flash` is retired (404), and
`gemini-3.7-flash` and the `gemini-flash-latest` alias both hang past 45s on the
free tier. `gemini-3.6-flash` answers in ~3s. Expect ~20-25s for a full
two-tool turn, which is three model calls.
