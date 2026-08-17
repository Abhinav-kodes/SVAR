# Phase 2: WebSocket Real-Time Progress Design

**Date:** 2026-08-17

## Problem

The dashboard polls `GET /api/progress?file=X` every 1.5 seconds while a
pipeline job runs. Polling adds network overhead, introduces up to 1.5s of
latency per stage transition, and keeps a request stream alive that could be
pushed instead. Phase 2 replaces polling with event-driven push: the worker
publishes stage transitions to Redis Pub/Sub, FastAPI pushes them to the React
dashboard over a WebSocket.

## Goals

- Replace the frontend 1.5s polling loop with a WebSocket push channel.
- Keep `GET /api/progress` working unchanged (backward compatibility, tests,
  external tools).
- Publishing must never fail a pipeline stage (publish errors are swallowed and
  logged).
- WebSocket connections must receive an immediate snapshot on connect so the
  UI is never blank, and must close cleanly once a job reaches a terminal state.

## Non-Goals

- No in-stage progress (e.g. percent within STT). Only stage transitions:
  `queued`, `running`, `done`, `error`, `completed`.
- No multi-client fan-out logic beyond per-file channels.
- No changes to `GET /api/progress` or its tests.

## Architecture

```
worker (pipeline/runner.py)
  └── ProgressPublisher.publish(filename, snapshot)
        └── Redis Pub/Sub channel: svar:progress:<filename>
              └── FastAPI /ws/progress (threadpool + blocking pubsub)
                    └── WebSocket frames (JSON progress snapshots)
                          └── React dashboard (App.tsx)
```

The progress snapshot published is always the live result of
`job_store.get(filename)` — the exact dict `GET /api/progress` returns today —
so the worker, the WebSocket, and the legacy endpoint all speak one shape.

## Components

### 1. `pipeline/progress_pubsub.py` (new module)

- `class ProgressPublisher` — ABC with `publish(self, filename: str, progress: dict) -> None`.
- `class RedisProgressPublisher(ProgressPublisher)`:
  - `__init__(self, url: str = "redis://localhost:6379/0")` — lazily imports
    `redis`, creates `redis.Redis.from_url(url)`.
  - `publish` — `self._redis.publish(f"svar:progress:{filename}", json.dumps(progress))`
    wrapped in `try/except Exception` that logs and swallows (never raises).
- `class NoopProgressPublisher(ProgressPublisher)` — `publish` does nothing.
  Used in tests and as a safe default.

No google imports, no RQ imports — pure pub/sub transport.

### 2. `pipeline/runner.py` (modified)

- `_timed_stage(filename, stage_id, job_store, fn)` gains a `publisher:
  ProgressPublisher` parameter. After each `update_stage` call (running, done,
  error), it publishes `job_store.get(filename)`.
- `run_pipeline(filename, job_store=None, results_repo=None, publisher=None)`:
  - default `publisher` is `RedisProgressPublisher(REDIS_URL)`
  - publishes after `job_store.create(filename)` (queued snapshot)
  - publishes after each `_timed_stage`
  - publishes after `job_store.finish(...)` (completed/error terminal snapshot)
- Callers (worker, tests) may pass `NoopProgressPublisher()` to silence events.

### 3. `api/main.py` (modified)

- `create_app(job_store, results_repo, enqueue, job_alive)` gains an optional
  `subscriber` dependency: `subscriber(filename: str) -> Subscription` where
  `Subscription` is a thin object exposing a blocking
  `get_message(timeout=None)` (matching redis-py pubsub semantics) and
  `close()`. Default implementation subscribes with redis-py to
  `svar:progress:<filename>`. Tests inject a fake.

- New route:

```python
@app.websocket("/ws/progress")
async def ws_progress(websocket: WebSocket, file: str = ""):
    await websocket.accept()
    snapshot = job_store.get(file)
    if snapshot is not None:
        await websocket.send_text(json.dumps(snapshot))
        if snapshot["status"] in ("completed", "error"):
            await websocket.close()
            return
    loop = asyncio.get_running_loop()
    sub = subscriber(file)
    try:
        while True:
            msg = await loop.run_in_executor(None, sub.get_message)
            if msg and msg.get("type") == "message":
                data = msg["data"]  # already a JSON string
                await websocket.send_text(data)
                if json.loads(data)["status"] in ("completed", "error"):
                    break
    except WebSocketDisconnect:
        pass
    finally:
        sub.close()
        try:
            await websocket.close()
        except RuntimeError:
            pass
```

- The blocking `get_message` runs in the default threadpool executor; no new
  dependencies (redis-py remains the single Redis client).

### 4. `dashboard-ui/src/App.tsx` (modified)

- Remove `pollProgress` callback and the `setInterval` poll effect.
- New effect keyed on `isAnalyzing && activeFile`:
  - opens `new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws/progress?file=${encodeURIComponent(activeFile)}`)`
  - `onmessage`: `JSON.parse` → `setProgress`; if `status === "completed"` →
    `setIsAnalyzing(false)` + `fetchAllResults(activeFile)`; if `status ===
    "error"` → `setIsAnalyzing(false)`.
  - `onclose`: reconnect after 1s with a timer unless the component is being
    torn down or a terminal state was already received.
  - cleanup: clear timer, close socket on unmount / file change.
- Rebuild with `npm run build` so `dashboard/dist` is updated.

### 5. Tests

- `pipeline/tests/test_progress_pubsub.py` (new):
  - stage transition publishes a snapshot with the file's current progress
    (recording publisher injected into `run_pipeline` with an in-memory
    job store)
  - terminal publish after `finish`
  - a publisher that raises does not fail the pipeline
  - `NoopProgressPublisher` performs no publish
- `api/tests/test_ws_progress.py` (new), using FastAPI TestClient
  `websocket_connect`:
  - connect sends the current snapshot immediately
  - connect with an already-completed job sends snapshot then closes
  - stage messages from the fake subscriber are forwarded verbatim
  - a terminal message closes the connection
  - client disconnect does not leak (subscriber closed)
- `api/tests/test_api.py` unchanged — `/api/progress` still passes.

### 6. Docs

- `Roadmap.md`: Phase 2 row → 🟢 100%, summary table row
  `| Phase 2: WebSockets (Pub/Sub) | 🟢 100% | Redis Pub/Sub → FastAPI → React push updates |`,
  Remaining Work item 4 marked ✅, remaining items renumbered (Phase 3 → 5,
  Phase 4 → 6).
- `README.md`: API table gains `| GET /ws/progress?file=X | WebSocket — live
  pipeline progress pushes |`; replace the "Progress polling is unchanged" note
  with a line describing WebSocket push (with `/api/progress` kept for
  compatibility).

## Error Handling

- **Publish failure** (Redis down in worker): caught, logged, stage proceeds.
- **Subscriber failure** (Redis down in API): the WebSocket closes with the
  exception; the frontend auto-reconnects after 1s.
- **Disconnect mid-stream**: `WebSocketDisconnect` caught, subscription closed.
- **Empty file param**: `job_store.get("")` returns `None` → no snapshot; the
  subscriber loop still runs for that channel (matches current `/api/progress`
  behavior of returning idle).

## Security

No auth exists on any endpoint today; the WebSocket follows the same model.
Only the sample-calls directory is reachable; the `file` param is
URL-encoded on the client and used verbatim as a channel name.
