# Phase 2 WebSocket Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dashboard's 1.5s polling of `/api/progress` with event-driven push: the worker publishes progress snapshots to Redis Pub/Sub, FastAPI forwards them over `/ws/progress`, and the React dashboard renders them live.

**Architecture:** worker → `ProgressPublisher.publish(filename, snapshot)` → Redis channel `svar:progress:<filename>` → FastAPI `/ws/progress` (blocking `get_message` in threadpool executor) → WebSocket JSON frames → React `App.tsx` (no polling, 1s auto-reconnect on drop). The snapshot is always the live `job_store.get(filename)` dict — the exact shape `GET /api/progress` returns.

**Tech Stack:** Python 3, FastAPI (TestClient for WS tests), redis-py Pub/Sub, React 18 + Vite (dashboard-ui/), RQ worker, pytest.

## Global Constraints

- `GET /api/progress` and `api/tests/test_api.py` stay unchanged — backward compatibility is a hard requirement.
- Publishing must never fail a pipeline stage: all publish errors are caught, logged, and swallowed.
- The `/ws/progress` handler forwards `msg["data"]` verbatim (it is already a JSON string from the worker).
- No new dependencies — redis-py remains the single Redis client; blocking reads run via `loop.run_in_executor(None, sub.get_message)`.
- Frontend source lives in `dashboard-ui/`; built output goes to `dashboard/dist` (vite `outDir: '../dashboard/dist'`); the API serves that dist. Rebuild with `npm run build` inside `dashboard-ui/`.
- Tests run with `venv/bin/python -m pytest <file> -q` from the repo root.

---

### Task 1: `pipeline/progress_pubsub.py` module + unit tests

**Files:**
- Create: `pipeline/progress_pubsub.py`
- Test: `pipeline/tests/test_progress_pubsub.py`

**Interfaces:**
- Produces: `ProgressPublisher` (ABC, `publish(self, filename: str, progress: dict) -> None`), `RedisProgressPublisher(ProgressPublisher)` (`__init__(self, url: str = "redis://localhost:6379/0")`, lazy `import redis`, `redis.Redis.from_url(url)`; `publish` sends `json.dumps(progress)` to channel `f"svar:progress:{filename}"`, wrapped in `try/except Exception` that logs and swallows), `NoopProgressPublisher(ProgressPublisher)` (`publish` does nothing).

- [ ] **Step 1: Write the failing tests**

```python
import json

import pytest

from pipeline.progress_pubsub import NoopProgressPublisher, RedisProgressPublisher


class FakeRedis:
    def __init__(self):
        self.published = []

    def publish(self, channel, message):
        self.published.append((channel, message))
        return 1


@pytest.fixture
def fake_redis(monkeypatch):
    import redis

    fake = FakeRedis()
    monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, url: fake))
    return fake


def test_publish_sends_json_to_per_file_channel(fake_redis):
    pub = RedisProgressPublisher("redis://test/0")
    pub.publish("a.mp3", {"status": "running", "percent": 11})
    assert fake_redis.published == [
        ("svar:progress:a.mp3", json.dumps({"status": "running", "percent": 11}))
    ]


def test_publish_swallows_redis_errors(fake_redis, caplog):
    def boom(channel, message):
        raise RuntimeError("redis down")

    fake_redis.publish = boom
    pub = RedisProgressPublisher("redis://test/0")
    pub.publish("a.mp3", {"status": "running"})  # must not raise
    assert "redis down" in caplog.text


def test_noop_publisher_does_nothing():
    pub = NoopProgressPublisher()
    pub.publish("a.mp3", {"status": "running"})  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest pipeline/tests/test_progress_pubsub.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.progress_pubsub'`

- [ ] **Step 3: Write the minimal implementation**

```python
import json
import logging

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ProgressPublisher(ABC):
    @abstractmethod
    def publish(self, filename: str, progress: dict) -> None:
        raise NotImplementedError


class RedisProgressPublisher(ProgressPublisher):
    def __init__(self, url: str = "redis://localhost:6379/0"):
        import redis

        self._redis = redis.Redis.from_url(url)

    def publish(self, filename: str, progress: dict) -> None:
        try:
            self._redis.publish(f"svar:progress:{filename}", json.dumps(progress))
        except Exception as e:
            logger.warning("progress publish failed for %s: %s", filename, e)


class NoopProgressPublisher(ProgressPublisher):
    def publish(self, filename: str, progress: dict) -> None:
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest pipeline/tests/test_progress_pubsub.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add pipeline/progress_pubsub.py pipeline/tests/test_progress_pubsub.py
git commit -m "feat: add progress publisher for Redis pub/sub"
```

---

### Task 2: Publish snapshots from `pipeline/runner.py`

**Files:**
- Modify: `pipeline/runner.py:12-23` (`_timed_stage`), `pipeline/runner.py:25-73` (`run_pipeline`)
- Modify: `pipeline/tests/test_runner.py` (3 existing tests gain `publisher=NoopProgressPublisher()`)
- Test: `pipeline/tests/test_progress_pubsub.py` (append 4 runner-level tests)

**Interfaces:**
- Consumes: `ProgressPublisher`, `RedisProgressPublisher`, `NoopProgressPublisher` from Task 1.
- Produces: `_timed_stage(filename, stage_id, job_store, publisher, fn)` — publishes `job_store.get(filename)` after each `update_stage` (running, done, error). `run_pipeline(filename, job_store=None, results_repo=None, publisher=None)` — default `RedisProgressPublisher(REDIS_URL)`; publishes after `job_store.create` (queued), after each `_timed_stage`, and after `job_store.finish` (completed/error terminal snapshot).

- [ ] **Step 1: Write the failing tests (append to test_progress_pubsub.py)**

```python
import pytest

from pipeline import runner, stages
from pipeline.job_store import InMemoryJobStore, STAGES
from pipeline.progress_pubsub import NoopProgressPublisher
from pipeline.results_repo import InMemoryResultsRepository


class RecordingPublisher:
    def __init__(self):
        self.published = []

    def publish(self, filename, progress):
        self.published.append((filename, dict(progress)))


class RaisingPublisher:
    def publish(self, filename, progress):
        raise RuntimeError("publish boom")


def _mock_stages(monkeypatch, tmp_path, fail_at=None):
    monkeypatch.setattr(runner, "SAMPLE_CALLS_DIR", str(tmp_path))
    for sid in [s["id"] for s in STAGES]:
        def make_stage(sid=sid):
            def stage(ctx):
                if fail_at and sid == fail_at:
                    raise ValueError(f"stage {sid} failed")
                ctx.cache[sid] = "done"
            return stage
        monkeypatch.setattr(stages, STAGE_FN.get(sid, f"stage_{sid}"), make_stage())
    monkeypatch.setattr(stages, "free_gpu", lambda label="": None)


def test_run_pipeline_publishes_snapshot_per_transition(monkeypatch, tmp_path):
    job_store = InMemoryJobStore()
    pub = RecordingPublisher()
    _mock_stages(monkeypatch, tmp_path)
    (tmp_path / "x.wav").write_bytes(b"not really audio")

    runner.run_pipeline("x.wav", job_store=job_store,
                        results_repo=InMemoryResultsRepository(), publisher=pub)

    filenames = [f for f, _ in pub.published]
    assert filenames == ["x.wav"] * len(pub.published)
    statuses = [p["status"] for _, p in pub.published]
    assert statuses[0] == "queued"
    assert "running" in statuses
    assert statuses[-1] == "completed"
    for _, p in pub.published:
        assert set(p["stages"]) == {s["id"] for s in STAGES}


def test_run_pipeline_publishes_terminal_error_snapshot(monkeypatch, tmp_path):
    job_store = InMemoryJobStore()
    pub = RecordingPublisher()
    _mock_stages(monkeypatch, tmp_path, fail_at="crm")
    (tmp_path / "x.wav").write_bytes(b"not really audio")

    with pytest.raises(ValueError, match="stage crm failed"):
        runner.run_pipeline("x.wav", job_store=job_store,
                            results_repo=InMemoryResultsRepository(), publisher=pub)

    assert pub.published[-1][1]["status"] == "error"


def test_raising_publisher_does_not_fail_pipeline(monkeypatch, tmp_path):
    job_store = InMemoryJobStore()
    _mock_stages(monkeypatch, tmp_path)
    (tmp_path / "x.wav").write_bytes(b"not really audio")

    runner.run_pipeline("x.wav", job_store=job_store,
                        results_repo=InMemoryResultsRepository(),
                        publisher=RaisingPublisher())

    assert job_store.get("x.wav")["status"] == "completed"


def test_noop_publisher_completes_silently(monkeypatch, tmp_path):
    job_store = InMemoryJobStore()
    _mock_stages(monkeypatch, tmp_path)
    (tmp_path / "x.wav").write_bytes(b"not really audio")

    runner.run_pipeline("x.wav", job_store=job_store,
                        results_repo=InMemoryResultsRepository(),
                        publisher=NoopProgressPublisher())

    assert job_store.get("x.wav")["status"] == "completed"
```

Note: `STAGE_FN` (mapping `text_emo`→`stage_text_emotion`, `compliance`→`stage_audit`) already exists in `pipeline/tests/test_runner.py`; add the same mapping at the top of `test_progress_pubsub.py` since tests there mock stages too:

```python
STAGE_FN = {"text_emo": "stage_text_emotion", "compliance": "stage_audit"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest pipeline/tests/test_progress_pubsub.py -q`
Expected: FAIL — `TypeError: run_pipeline() got an unexpected keyword argument 'publisher'` (publisher not yet threaded through).

- [ ] **Step 3: Implement publisher wiring in runner.py**

Replace `_timed_stage` (lines 12-22):

```python
def _timed_stage(filename: str, stage_id: str, job_store: JobStore, publisher, fn):
    job_store.update_stage(filename, stage_id, "running")
    publisher.publish(filename, job_store.get(filename))
    t0 = time.time()
    try:
        result = fn()
        elapsed = round(time.time() - t0, 2)
        job_store.update_stage(filename, stage_id, "done", time_s=elapsed)
        publisher.publish(filename, job_store.get(filename))
        return result
    except Exception:
        job_store.update_stage(filename, stage_id, "error")
        publisher.publish(filename, job_store.get(filename))
        raise
```

Update `run_pipeline` (lines 25-73): import `RedisProgressPublisher` and `REDIS_URL`, add the `publisher` parameter and default, publish after `create`, after `finish` in both success and error paths, and pass `publisher` to every `_timed_stage` call:

```python
from api.config import REDIS_URL, SAMPLE_CALLS_DIR
...
from pipeline.progress_pubsub import RedisProgressPublisher


def run_pipeline(
    filename: str,
    job_store: Optional[JobStore] = None,
    results_repo: Optional[ResultsRepository] = None,
    publisher=None,
) -> dict:
    job_store = job_store or RedisJobStore()
    results_repo = results_repo or PostgresResultsRepository()
    publisher = publisher or RedisProgressPublisher(REDIS_URL)
    ...
    job_store.create(filename)
    publisher.publish(filename, job_store.get(filename))
    t0 = time.time()
    try:
        _timed_stage(filename, "denoise", job_store, publisher, lambda: stages.stage_denoise(ctx))
        _timed_stage(filename, "diarize", job_store, publisher, lambda: stages.stage_diarize(ctx))
        _timed_stage(filename, "stt", job_store, publisher, lambda: stages.stage_stt(ctx))
        _timed_stage(filename, "acoustic", job_store, publisher, lambda: stages.stage_acoustic(ctx))
        free_gpu("stt+acoustic done")

        _timed_stage(filename, "text_emo", job_store, publisher, lambda: stages.stage_text_emotion(ctx))
        free_gpu("text_emo done")

        _timed_stage(filename, "compliance", job_store, publisher, lambda: stages.stage_audit(ctx))
        _timed_stage(filename, "fusion", job_store, publisher, lambda: stages.stage_fusion(ctx))
        _timed_stage(filename, "qa", job_store, publisher, lambda: stages.stage_qa(ctx))
        _timed_stage(filename, "crm", job_store, publisher, lambda: stages.stage_crm(ctx))
        ...
        results_repo.save(filename, results)
        job_store.finish(filename)
        publisher.publish(filename, job_store.get(filename))
        return results
    except Exception as e:
        job_store.finish(filename, error=str(e))
        publisher.publish(filename, job_store.get(filename))
        raise
```

- [ ] **Step 4: Update the 3 existing tests in `pipeline/tests/test_runner.py`**

Each `runner.run_pipeline(...)` call in that file (lines 17, 36, 59) gains `publisher=NoopProgressPublisher()` so unit tests never touch Redis:

```python
runner.run_pipeline("nope.mp3", job_store=job_store, results_repo=repo, publisher=NoopProgressPublisher())
```

Add `from pipeline.progress_pubsub import NoopProgressPublisher` to the imports.

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/python -m pytest pipeline/tests/test_progress_pubsub.py pipeline/tests/test_runner.py -q`
Expected: PASS (7 + 3 tests)

- [ ] **Step 6: Commit**

```bash
git add pipeline/runner.py pipeline/tests/test_runner.py pipeline/tests/test_progress_pubsub.py
git commit -m "feat: publish progress snapshots to redis pub/sub from runner"
```

---

### Task 3: `/ws/progress` WebSocket endpoint in `api/main.py`

**Files:**
- Modify: `api/main.py` — imports (lines 1-19), `create_app` signature (lines 99-113), new route after `/api/progress` (line 174)
- Test: `api/tests/test_ws_progress.py` (new)

**Interfaces:**
- Consumes: nothing from Tasks 1-2 (subscriber is its own dependency).
- Produces: `create_app(..., subscriber=None)` where `subscriber(filename: str) -> Subscription`; `Subscription` exposes blocking `get_message(timeout=None)` (redis-py pubsub semantics) and `close()`. Default `_default_subscriber(filename)` returns `redis.Redis.from_url(REDIS_URL).pubsub()` after `subscribe(f"svar:progress:{filename}")`. New route `GET /ws/progress?file=X` (WebSocket).

- [ ] **Step 1: Write the failing tests**

```python
import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from api.main import create_app
from pipeline.job_store import InMemoryJobStore
from pipeline.results_repo import InMemoryResultsRepository


class FakeSubscription:
    def __init__(self, messages=None):
        self._queue = list(messages or [])
        self.received = []
        self.closed = False

    def get_message(self, timeout=None):
        if self._queue:
            msg = self._queue.pop(0)
            self.received.append(msg)
            return msg
        return None

    def close(self):
        self.closed = True


def make_app(sub, job_store=None):
    return create_app(
        job_store=job_store or InMemoryJobStore(),
        results_repo=InMemoryResultsRepository(),
        enqueue=lambda f: None,
        job_alive=lambda f: True,
        subscriber=lambda filename: sub,
    )


def test_ws_sends_current_snapshot_on_connect():
    job_store = InMemoryJobStore()
    job_store.create("a.mp3")
    app = make_app(FakeSubscription(), job_store=job_store)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/progress?file=a.mp3") as ws:
            data = json.loads(ws.receive_text())
            assert data["status"] == "queued"
            assert "stages" in data


def test_ws_completed_job_snapshot_then_close():
    job_store = InMemoryJobStore()
    job_store.create("a.mp3")
    job_store.finish("a.mp3")
    app = make_app(FakeSubscription(), job_store=job_store)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/progress?file=a.mp3") as ws:
            data = json.loads(ws.receive_text())
            assert data["status"] == "completed"
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()


def test_ws_forwards_messages_verbatim_and_closes_on_terminal():
    sub = FakeSubscription(messages=[
        {"type": "message", "data": json.dumps({"status": "running", "percent": 50, "stages": {}})},
        {"type": "message", "data": json.dumps({"status": "completed", "percent": 100, "stages": {}})},
    ])
    app = make_app(sub)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/progress?file=a.mp3") as ws:
            assert json.loads(ws.receive_text())["status"] == "running"
            assert json.loads(ws.receive_text())["status"] == "completed"
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()

    assert sub.closed
    assert len(sub.received) == 2


def test_ws_client_disconnect_closes_subscriber():
    sub = FakeSubscription()
    app = make_app(sub)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/progress?file=a.mp3"):
            pass  # nothing is sent (no snapshot, empty queue); just connect and disconnect

    assert sub.closed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest api/tests/test_ws_progress.py -q`
Expected: FAIL — the connection closes without accepting (no `/ws/progress` route yet); the exact exception varies by starlette version (e.g. `WebSocketDisconnect`).

- [ ] **Step 3: Implement the subscriber default and WebSocket route**

Update imports in `api/main.py`:

```python
import asyncio
import json
import os
import urllib.parse

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from starlette.websockets import WebSocketDisconnect
```

Add `_default_subscriber` near `_default_job_alive` (after line 83):

```python
def _default_subscriber(filename: str):
    import redis

    conn = redis.Redis.from_url(REDIS_URL)
    sub = conn.pubsub()
    sub.subscribe(f"svar:progress:{filename}")
    return sub
```

Extend `create_app` (lines 99-108):

```python
def create_app(
    job_store: JobStore = None,
    results_repo: ResultsRepository = None,
    enqueue=None,
    job_alive=None,
    subscriber=None,
) -> FastAPI:
    job_store = job_store or RedisJobStore(REDIS_URL)
    results_repo = results_repo or _LazyResultsRepository()
    enqueue = enqueue or _default_enqueue
    job_alive = job_alive or _default_job_alive
    subscriber = subscriber or _default_subscriber
```

Add the WebSocket route immediately after the `/api/progress` route (after line 173):

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

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest api/tests/test_ws_progress.py -q`
Expected: PASS (4 passed). If a test hangs: the FakeSubscription's non-blocking `get_message` (returns `None` with an empty queue) keeps the loop spinning in the executor — that is expected; the loop only exits via client disconnect or a terminal message. If `pytest.raises(WebSocketDisconnect)` raises a different exception type in your environment, record the deviation and use the actual type.

- [ ] **Step 5: Verify `/api/progress` still passes unchanged**

Run: `venv/bin/python -m pytest api/tests/test_api.py -q`
Expected: PASS (all existing tests, no modifications to the file)

- [ ] **Step 6: Commit**

```bash
git add api/main.py api/tests/test_ws_progress.py
git commit -m "feat: add /ws/progress websocket endpoint for live progress push"
```

---

### Task 4: React dashboard consumes WebSocket instead of polling

**Files:**
- Modify: `dashboard-ui/src/App.tsx` (remove lines 60-94; add WS effect)
- Rebuild: `dashboard-ui/` → `dashboard/dist` via `npm run build`

**Interfaces:**
- Consumes: `/ws/progress?file=<encodeURIComponent(activeFile)>` from Task 3; `ProgressState` type from `./types/dashboard`.

- [ ] **Step 1: Replace `pollProgress` + poll effect with a WebSocket effect**

Delete the `pollProgress` useCallback (lines 60-80) and the poll loop effect (lines 82-94). Add this effect in their place:

```tsx
  // Live progress over WebSocket (replaces 1.5s polling)
  useEffect(() => {
    if (!isAnalyzing || !activeFile) return;
    let ws: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let tornDown = false;
    let receivedTerminal = false;

    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      ws = new WebSocket(`${proto}//${location.host}/ws/progress?file=${encodeURIComponent(activeFile)}`);
      ws.onmessage = (ev) => {
        try {
          const p: ProgressState = JSON.parse(ev.data);
          setProgress(p);
          if (p.status === 'completed' || p.status === 'error') {
            receivedTerminal = true;
            setIsAnalyzing(false);
            if (p.status === 'completed') {
              fetchAllResults(activeFile);
            }
          }
        } catch (err) {
          console.error('Error parsing progress message:', err);
        }
      };
      ws.onclose = () => {
        if (!tornDown && !receivedTerminal) {
          timer = setTimeout(connect, 1000);
        }
      };
    };
    connect();

    return () => {
      tornDown = true;
      if (timer) clearTimeout(timer);
      if (ws) ws.close();
    };
  }, [isAnalyzing, activeFile, fetchAllResults]);
```

- [ ] **Step 2: Build the dashboard**

Run: `cd dashboard-ui && npm run build`
Expected: `tsc -b` passes and vite emits to `../dashboard/dist` (no TypeScript errors).

- [ ] **Step 3: Verify the frontend still type-checks and the diff is only the expected changes**

Run: `git diff --stat dashboard-ui/src/App.tsx dashboard/dist | tail -5`
Expected: `App.tsx` shows deletions of the poll code and the new effect; `dashboard/dist` shows rebuilt assets.

- [ ] **Step 4: Commit**

```bash
git add dashboard-ui/src/App.tsx dashboard/dist
git commit -m "feat: replace progress polling with websocket push in dashboard"
```

---

### Task 5: Docs + full suite verification

**Files:**
- Modify: `Roadmap.md` (Phase 2 row, summary table, Remaining Work list)
- Modify: `README.md` (API table, polling note)
- Verify: full test suite

- [ ] **Step 1: Update `Roadmap.md`**

- Phase 2 row (Remaining Work item 4 area): change `| 4. Phase 2 — WebSocket real-time updates (replaces /api/progress polling) | 🔴 0% |` to `| ~~4. Phase 2 — WebSocket real-time updates (replaces /api/progress polling)~~ | ✅ 100% | ... |` matching the file's existing ✅-item formatting.
- Summary table: add/update row `| Phase 2: WebSockets (Pub/Sub) | 🟢 100% | Redis Pub/Sub → FastAPI → React push updates |`.
- Remaining Work: mark item 4 ✅ Complete, renumber the rest (Phase 3 → 5, Phase 4 → 6).

- [ ] **Step 2: Update `README.md`**

- API table: add row `| GET /ws/progress?file=X | WebSocket — live pipeline progress pushes |` (place near the `/api/progress` row).
- Replace the "Progress polling is unchanged" note (~line 120) with a line describing WebSocket push: the worker publishes stage transitions to Redis Pub/Sub, the dashboard receives them live over `/ws/progress`, and `/api/progress` is kept for compatibility.

- [ ] **Step 3: Run the full test suite**

Run: `venv/bin/python -m pytest -q`
Expected: previous baseline 80 passed / 3 skipped / 4 failed + 4 ws tests + 7 pubsub/runner tests = **91 passed, 3 skipped, 4 failed** (the 4 failures are the pre-existing, unrelated ones: `diarization/tests/test_pipeline.py::TestDiarizationPipeline::test_diarization_pipeline_execution` and 3× `sentiment/tests/test_stt_transcriber.py` method-drift tests).

- [ ] **Step 4: Commit**

```bash
git add Roadmap.md README.md
git commit -m "docs: mark Phase 2 websocket progress complete"
```