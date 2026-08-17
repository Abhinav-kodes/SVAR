# FastAPI Backend (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-process `dashboard/dashboard_server.py` (http.server + background thread) with a distributed Phase-1 backend: FastAPI API server + Redis job queue with a separate RQ worker process + PostgreSQL result persistence — without changing the frontend or the ML pipeline.

**Architecture:** The FastAPI app keeps the exact same API contract the React frontend already uses (`/api/sample_calls`, `/api/analyze`, `/api/progress`, `/api/results`, `/audio/<file>`, static dashboard). On analyze it writes a job to Redis and enqueues an RQ job; a separate worker process executes the existing 9 pipeline stages (moved verbatim into `pipeline/stages.py`), updating progress in Redis and persisting final results to Postgres. `/api/progress` and `/api/results` read from Redis/Postgres, so the API stays responsive and survives restarts.

**Tech Stack:** FastAPI + uvicorn, redis-py + RQ, psycopg (v3), Pydantic v2, pytest + httpx TestClient. New packages: `fastapi`, `uvicorn`, `redis`, `rq`, `psycopg[binary]`, `httpx`, `pytest`.

## Global Constraints

- **Frontend untouched.** The 5 endpoints + JSON shapes the React app uses (`dashboard-ui/src/App.tsx`) must work unchanged: `GET /api/sample_calls` → `string[]`; `POST /api/analyze` `{filename}` → `{status, message}`; `GET /api/progress?file=<filename>` → progress dict; `POST /api/results` `{filename}` → `CallData` dict; `GET /audio/<file>` → bytes. Static: `/` → `dashboard/dist/index.html` (fallback `dashboard/index.html`), `/assets/*` → `dashboard/dist/assets/*`.
- **Progress JSON shape preserved exactly** (frontend `ProgressState`): `{status, current_stage, percent, stages: {<id>: {status, time_s}}, error, start_time, time_s}` with statuses `idle|queued|running|completed|error` and stage statuses `pending|running|done|error`. Stage ids: `denoise, diarize, stt, acoustic, text_emo, compliance, fusion, qa, crm`.
- **Pipeline stage order preserved** (from `_run_pipeline` in dashboard_server.py:411-442): `denoise → diarize → stt → acoustic → text_emo → compliance (runs stage_audit) → fusion → qa → crm`. GPU freed between stages (models unloaded + `torch.cuda.empty_cache()`).
- **No ML/pipeline logic changes.** Stages are moved verbatim from `dashboard/dashboard_server.py` to `pipeline/stages.py` with only the cache/globals plumbing refactored (see Task 3 transformation rule).
- **One worker process per GPU.** RQ worker runs in its own process; stage singleton models are process-locals.
- **Sample-call scope.** Phase 1 analyzes files already in `data/sample_calls/` (no upload endpoint; roadmap upload flow is future work). `POST /api/analyze` validates the file exists and returns 404 `{"error": ...}` otherwise (matches legacy behavior).
- **Repo conventions:** modules importable from repo root (existing style: `from denoising.pipeline import ...`); tests live in `<package>/tests/` (existing convention); commit messages in conventional style (`feat:`, `refactor:`, `docs:`, `test:` — see `git log --oneline`).
- **Config via env vars** with sensible local defaults (no .env file parsing in Phase 1): see Task 1.

## File Structure

- `requirements.txt` — consolidated dependency list (existing manual installs + new packages)
- `api/__init__.py`, `api/config.py` — env config (port, redis, postgres, paths)
- `api/schemas.py` — Pydantic request models
- `api/main.py` — FastAPI app factory + all routes + static/audio serving
- `api/tests/test_api.py` — endpoint tests via TestClient with in-memory stores
- `pipeline/__init__.py`
- `pipeline/job_store.py` — `JobStore` interface, `RedisJobStore`, `InMemoryJobStore`, `STAGES`, progress helpers
- `pipeline/results_repo.py` — `ResultsRepository` interface, `PostgresResultsRepository`, `InMemoryResultsRepository`
- `pipeline/stages.py` — the 9 stage functions moved from `dashboard/dashboard_server.py`, refactored to `JobContext`
- `pipeline/runner.py` — `run_pipeline()` orchestration (stage order, timing, GPU frees, result persistence)
- `pipeline/worker.py` — RQ worker entrypoint (`python -m pipeline.worker`)
- `pipeline/tests/test_job_store.py`, `pipeline/tests/test_results_repo.py`, `pipeline/tests/test_runner.py`, `pipeline/tests/test_stages.py`

Deleted at the end (Task 8): `dashboard/dashboard_server.py`.

---

### Task 1: Project scaffolding — requirements.txt + api/config.py

**Files:**
- Create: `requirements.txt`
- Create: `api/__init__.py` (empty)
- Create: `api/config.py`
- Test: `api/tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `api/config.py` constants: `PORT: int`, `REDIS_URL: str`, `DATABASE_URL: str`, `REPO_ROOT: str`, `SAMPLE_CALLS_DIR: str`, `DASHBOARD_DIR: str` — consumed by all later tasks.

- [ ] **Step 1: Write the failing test**

`api/tests/test_config.py`:

```python
import importlib
import sys


def test_config_defaults(monkeypatch):
    for var in ("SVAR_PORT", "SVAR_REDIS_URL", "SVAR_DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)
    sys.modules.pop("api.config", None)
    cfg = importlib.import_module("api.config")
    assert cfg.PORT == 8050
    assert cfg.REDIS_URL == "redis://localhost:6379/0"
    assert cfg.DATABASE_URL == "postgresql://svar:svar@localhost:5432/svar"
    assert SAMPLE_CALLS_DIR.endswith(("sample_calls", "sample_calls/"))


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("SVAR_PORT", "9000")
    monkeypatch.setenv("SVAR_REDIS_URL", "redis://example:6390/2")
    sys.modules.pop("api.config", None)
    cfg = importlib.import_module("api.config")
    assert cfg.PORT == 9000
    assert cfg.REDIS_URL == "redis://example:6390/2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest api/tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.config'` (and `SAMPLE_CALLS_DIR` NameError)

- [ ] **Step 3: Write minimal implementation**

`api/config.py`:

```python
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SAMPLE_CALLS_DIR = os.path.join(REPO_ROOT, "data", "sample_calls")
DASHBOARD_DIR = os.path.join(REPO_ROOT, "dashboard")

PORT = int(os.getenv("SVAR_PORT", "8050"))
REDIS_URL = os.getenv("SVAR_REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL = os.getenv("SVAR_DATABASE_URL", "postgresql://svar:svar@localhost:5432/svar")
```

`requirements.txt` (consolidated — existing installs plus Phase-1 additions):

```text
--extra-index-url https://download.pytorch.org/whl/cu128
torch
torchaudio
torchvision
pyannote.audio
speechbrain
librosa
soundfile
numpy
scikit-learn
joblib
google-cloud-speech
transformers
sentencepiece
pyyaml
fastapi
uvicorn
redis
rq
psycopg[binary]
httpx
pytest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest api/tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt api/__init__.py api/config.py api/tests/test_config.py
git commit -m "feat: scaffold FastAPI backend with env config and consolidated requirements"
```

---

### Task 2: JobStore — progress state in Redis (with in-memory twin for tests)

**Files:**
- Create: `pipeline/__init__.py` (empty)
- Create: `pipeline/job_store.py`
- Test: `pipeline/tests/test_job_store.py`

**Interfaces:**
- Consumes: nothing (standalone)
- Produces — consumed by Tasks 3, 5, 6, 7:
  - `STAGES: list[dict]` — same 9 entries as dashboard_server.py:24-34 (id + label)
  - `class JobStore` with:
    - `create(filename: str) -> None` — initializes `queued` progress dict
    - `update_stage(filename: str, stage_id: str, status: str, time_s: float = 0.0) -> None` — `status` in `pending|running|done|error`; recomputes `percent`
    - `finish(filename: str, error: str | None = None) -> None` — sets `completed`/`error`, `time_s`
    - `get(filename: str) -> dict | None` — returns the progress dict (default `{"status": "idle", "percent": 0, "stages": {}}` if absent? No — return `None`; caller handles default)
  - `class InMemoryJobStore(JobStore)` — dict + `threading.Lock`, same key semantics as Redis impl (used in tests and as fallback)
  - `class RedisJobStore(JobStore)` — `__init__(self, url: str = "redis://localhost:6379/0")`, lazy `redis.Redis.from_url`, key `svar:job:{filename}`, value JSON, TTL 86400

- [ ] **Step 1: Write the failing test**

`pipeline/tests/test_job_store.py` (runs against `InMemoryJobStore` — no Redis needed; `RedisJobStore` gets the same assertions via a shared helper):

```python
import pytest

from pipeline.job_store import InMemoryJobStore, STAGES


def make_store():
    return InMemoryJobStore()


def test_create_sets_queued_progress():
    store = make_store()
    store.create("a.mp3")
    p = store.get("a.mp3")
    assert p["status"] == "queued"
    assert p["percent"] == 0
    assert set(p["stages"].keys()) == {s["id"] for s in STAGES}


def test_update_stage_running_and_done():
    store = make_store()
    store.create("a.mp3")
    store.update_stage("a.mp3", "denoise", "running")
    p = store.get("a.mp3")
    assert p["current_stage"] == "denoise"
    assert p["stages"]["denoise"]["status"] == "running"
    store.update_stage("a.mp3", "denoise", "done", time_s=1.5)
    p = store.get("a.mp3")
    assert p["stages"]["denoise"]["status"] == "done"
    assert p["stages"]["denoise"]["time_s"] == 1.5
    assert p["percent"] == round(1 / len(STAGES) * 100)


def test_finish_completed_sets_time_s():
    store = make_store()
    store.create("a.mp3")
    store.finish("a.mp3")
    p = store.get("a.mp3")
    assert p["status"] == "completed"
    assert p["percent"] == 100
    assert p["time_s"] >= 0


def test_finish_error():
    store = make_store()
    store.create("a.mp3")
    store.finish("a.mp3", error="boom")
    p = store.get("a.mp3")
    assert p["status"] == "error"
    assert p["error"] == "boom"


def test_get_missing_returns_none():
    assert make_store().get("nope.mp3") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pipeline/tests/test_job_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.job_store'`

- [ ] **Step 3: Write minimal implementation**

`pipeline/job_store.py`:

```python
import json
import threading
import time
from typing import Dict, Optional

STAGES = [
    {"id": "denoise", "label": "Denoising"},
    {"id": "diarize", "label": "Diarization"},
    {"id": "stt", "label": "Speech-to-Text"},
    {"id": "acoustic", "label": "Acoustic Emotion"},
    {"id": "text_emo", "label": "Text Emotion"},
    {"id": "compliance", "label": "Compliance"},
    {"id": "fusion", "label": "Emotion Fusion"},
    {"id": "qa", "label": "QA Scoring"},
    {"id": "crm", "label": "CRM Note"},
]


def _new_progress() -> dict:
    return {
        "status": "queued",
        "current_stage": "",
        "percent": 0,
        "stages": {s["id"]: {"status": "pending", "time_s": 0} for s in STAGES},
        "error": None,
        "start_time": time.time(),
    }


class JobStore:
    def create(self, filename: str) -> None:
        raise NotImplementedError

    def update_stage(self, filename: str, stage_id: str, status: str, time_s: float = 0.0) -> None:
        raise NotImplementedError

    def finish(self, filename: str, error: Optional[str] = None) -> None:
        raise NotImplementedError

    def get(self, filename: str) -> Optional[dict]:
        raise NotImplementedError


class InMemoryJobStore(JobStore):
    def __init__(self):
        self._jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, filename: str) -> None:
        with self._lock:
            self._jobs[filename] = _new_progress()

    def update_stage(self, filename: str, stage_id: str, status: str, time_s: float = 0.0) -> None:
        with self._lock:
            p = self._jobs.get(filename)
            if not p:
                return
            p["current_stage"] = stage_id
            if status == "done":
                p["stages"][stage_id]["status"] = "done"
                p["stages"][stage_id]["time_s"] = time_s
            elif status == "error":
                p["stages"][stage_id]["status"] = "error"
            else:
                p["stages"][stage_id]["status"] = "running"
            done = sum(1 for s in p["stages"].values() if s["status"] == "done")
            p["percent"] = round(done / len(STAGES) * 100)

    def finish(self, filename: str, error: Optional[str] = None) -> None:
        with self._lock:
            p = self._jobs.get(filename)
            if not p:
                return
            p["status"] = "error" if error else "completed"
            p["percent"] = 100 if not error else p["percent"]
            p["error"] = error
            p["time_s"] = round(time.time() - p["start_time"], 1)

    def get(self, filename: str) -> Optional[dict]:
        with self._lock:
            p = self._jobs.get(filename)
            return dict(p) if p else None


class RedisJobStore(JobStore):
    TTL_SECONDS = 86400

    def __init__(self, url: str = "redis://localhost:6379/0"):
        import redis
        self._redis = redis.Redis.from_url(url)
        self._lock = threading.Lock()

    def _key(self, filename: str) -> str:
        return f"svar:job:{filename}"

    def create(self, filename: str) -> None:
        self._redis.set(self._key(filename), json.dumps(_new_progress()), ex=self.TTL_SECONDS)

    def update_stage(self, filename: str, stage_id: str, status: str, time_s: float = 0.0) -> None:
        with self._lock:
            raw = self._redis.get(self._key(filename))
            if raw is None:
                return
            p = json.loads(raw)
            p["current_stage"] = stage_id
            if status == "done":
                p["stages"][stage_id]["status"] = "done"
                p["stages"][stage_id]["time_s"] = time_s
            elif status == "error":
                p["stages"][stage_id]["status"] = "error"
            else:
                p["stages"][stage_id]["status"] = "running"
            done = sum(1 for s in p["stages"].values() if s["status"] == "done")
            p["percent"] = round(done / len(STAGES) * 100)
            self._redis.set(self._key(filename), json.dumps(p), ex=self.TTL_SECONDS)

    def finish(self, filename: str, error: Optional[str] = None) -> None:
        raw = self._redis.get(self._key(filename))
        if raw is None:
            return
        p = json.loads(raw)
        p["status"] = "error" if error else "completed"
        p["percent"] = 100 if not error else p["percent"]
        p["error"] = error
        p["time_s"] = round(time.time() - p["start_time"], 1)
        self._redis.set(self._key(filename), json.dumps(p), ex=self.TTL_SECONDS)

    def get(self, filename: str) -> Optional[dict]:
        raw = self._redis.get(self._key(filename))
        return json.loads(raw) if raw else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest pipeline/tests/test_job_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/__init__.py pipeline/job_store.py pipeline/tests/test_job_store.py
git commit -m "feat: add JobStore for Redis-backed pipeline progress with in-memory twin"
```

---

### Task 3: Move pipeline stages into pipeline/stages.py with JobContext

**Files:**
- Create: `pipeline/stages.py`
- Test: `pipeline/tests/test_stages.py`

**Interfaces:**
- Consumes: `JobStore` + `STAGES` from Task 2; `api/config.SAMPLE_CALLS_DIR`
- Produces — consumed by Task 5 (`runner.py`):
  - `@dataclass class JobContext: filepath: str; filename: str; cache: dict; job_store: JobStore`
  - `def stage_denoise(ctx)`, `stage_diarize(ctx)`, `stage_stt(ctx)`, `stage_acoustic(ctx)`, `stage_text_emotion(ctx)`, `stage_compliance(ctx)`, `stage_fusion(ctx)`, `stage_qa(ctx)`, `stage_crm(ctx)`, `stage_audit(ctx)` — same behavior as the functions in `dashboard/dashboard_server.py:179-390`, with `cache` held on `ctx` instead of the global `_cache`
  - `def free_gpu(label: str = "") -> None` — moved from `dashboard_server.py:398-408`

**Transformation rule (verbatim move, mechanical substitutions only):** copy each `stage_*` function body from `dashboard/dashboard_server.py:179-390` and `_free_gpu` from lines 398-408 with exactly these substitutions:
1. Signature `def stage_X(filename)` → `def stage_X(ctx: JobContext)`
2. `c = _ensure_cache(filename)` → `c = ctx.cache`
3. `filepath = os.path.join(SAMPLE_CALLS_DIR, filename)` → `filepath = ctx.filepath` (stage_denoise only)
4. Inner calls `stage_X(filename)` → `stage_X(ctx)` (e.g. `stage_stt` calls `stage_diarize(filename)` → `stage_diarize(ctx)`)
5. `log(...)` → keep `log()` helper copied as-is (print with timestamp, from dashboard_server.py:47-49)
6. Module-level lazy singletons `_get_denoiser/_get_diarizer/_get_stt/_get_acoustic/_get_emotion_classifier/_get_role_engine` (dashboard_server.py:52-101) copied as-is — they are worker-process globals now (each worker process has its own)

- [ ] **Step 1: Write the failing test**

`pipeline/tests/test_stages.py`:

```python
import numpy as np
import pytest

from pipeline.job_store import InMemoryJobStore
from pipeline.stages import JobContext, stage_denoise


@pytest.fixture
def ctx(tmp_path):
    import scipy.io.wavfile as wavfile
    path = tmp_path / "synthetic.wav"
    sr = 16000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wavfile.write(str(path), sr, audio)
    return JobContext(filepath=str(path), filename="synthetic.wav", cache={}, job_store=InMemoryJobStore())


def test_stage_denoise_populates_cache(ctx):
    stage_denoise(ctx)
    assert "audio" in ctx.cache
    assert "clean_audio" in ctx.cache
    assert "denoise_metrics" in ctx.cache
    assert ctx.cache["sr"] == 16000
    assert ctx.cache["duration"] == pytest.approx(2.0, abs=0.1)
    assert len(ctx.cache["clean_audio"]) == len(ctx.cache["audio"])


def test_stage_denoise_idempotent(ctx):
    stage_denoise(ctx)
    first = ctx.cache["denoise_metrics"]
    stage_denoise(ctx)
    assert ctx.cache["denoise_metrics"] is first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pipeline/tests/test_stages.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.stages'`

- [ ] **Step 3: Write minimal implementation**

`pipeline/stages.py` — copy per the transformation rule above. Reference the source line ranges for each function:
- `log` — dashboard_server.py:47-49
- `_get_denoiser`, `_get_diarizer`, `_get_stt`, `_get_acoustic`, `_get_emotion_classifier`, `_get_role_engine` — dashboard_server.py:52-101 (copied verbatim; they are now worker-process singletons)
- `JobContext` + `free_gpu`:

```python
import gc
from dataclasses import dataclass, field
from typing import Any, Dict

from pipeline.job_store import JobStore


@dataclass
class JobContext:
    filepath: str
    filename: str
    cache: Dict[str, Any] = field(default_factory=dict)
    job_store: JobStore = None


def free_gpu(label: str = ""):
    global _role_engine, _emotion_classifier, _diarizer
    import torch
    if not torch.cuda.is_available():
        return
    _role_engine = None
    _emotion_classifier = None
    _diarizer = None
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  [gpu] freed memory ({label})")
```

- `stage_denoise` — dashboard_server.py:179-190, substituted per rule
- `stage_diarize` — dashboard_server.py:193-207 (note: it imports `DiarizationPipeline.offload_to_cpu()` inline — keep)
- `stage_stt` — dashboard_server.py:210-253 (Gemini role resolver + MuRIL fallback chain kept verbatim)
- `stage_acoustic` — dashboard_server.py:256-274
- `stage_text_emotion` — dashboard_server.py:277-294
- `stage_compliance` — dashboard_server.py:297-304
- `stage_fusion` — dashboard_server.py:307-327
- `stage_qa` — dashboard_server.py:330-347
- `stage_crm` — dashboard_server.py:351-359
- `stage_audit` — dashboard_server.py:363-390 (unified Gemini audit with per-field fallback to `stage_compliance`/`stage_qa`/`stage_crm`)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest pipeline/tests/test_stages.py -v`
Expected: PASS (denoise runs on CPU — no GPU needed for this synthetic test)

- [ ] **Step 5: Verify the moved module imports cleanly (all lazy imports resolve)**

Run: `python -c "import pipeline.stages"` and `python -c "from pipeline.stages import stage_audit, stage_denoise, free_gpu"`
Expected: no errors (network/GPU modules load lazily inside functions)

- [ ] **Step 6: Commit**

```bash
git add pipeline/stages.py pipeline/tests/test_stages.py
git commit -m "refactor: move pipeline stages from dashboard_server into pipeline/stages with JobContext"
```

---

### Task 4: ResultsRepository — Postgres persistence (with in-memory twin for tests)

**Files:**
- Create: `pipeline/results_repo.py`
- Test: `pipeline/tests/test_results_repo.py`

**Interfaces:**
- Consumes: `api/config.DATABASE_URL`
- Produces — consumed by Tasks 5, 6, 7:
  - `class ResultsRepository`:
    - `init_db() -> None`
    - `save(filename: str, results: dict) -> None` — upsert by filename
    - `get(filename: str) -> dict | None`
  - `class InMemoryResultsRepository(ResultsRepository)`
  - `class PostgresResultsRepository(ResultsRepository)` — `__init__(self, url: str = "postgresql://svar:svar@localhost:5432/svar")`, lazy `psycopg.connect`; table `calls (id BIGSERIAL PRIMARY KEY, filename TEXT UNIQUE NOT NULL, results JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now())`

- [ ] **Step 1: Write the failing test**

`pipeline/tests/test_results_repo.py`:

```python
import pytest

from pipeline.results_repo import InMemoryResultsRepository


def make_repo():
    return InMemoryResultsRepository()


def test_save_and_get_roundtrip():
    repo = make_repo()
    repo.save("a.mp3", {"qa": {"score": 70}, "crm_note": "hi"})
    res = repo.get("a.mp3")
    assert res["qa"]["score"] == 70
    assert res["crm_note"] == "hi"


def test_get_missing_returns_none():
    assert make_repo().get("nope.mp3") is None


def test_save_upserts():
    repo = make_repo()
    repo.save("a.mp3", {"v": 1})
    repo.save("a.mp3", {"v": 2})
    assert repo.get("a.mp3")["v"] == 2
```

(Add the same three assertions against `PostgresResultsRepository` in a test class skipped unless `SVAR_TEST_POSTGRES` env var is set, e.g. `pytest.mark.skipif(os.getenv("SVAR_TEST_POSTGRES") != "1", reason="needs postgres")` — the CI/本地 default run needs no Postgres.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pipeline/tests/test_results_repo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.results_repo'`

- [ ] **Step 3: Write minimal implementation**

`pipeline/results_repo.py`:

```python
import os
import threading
from typing import Dict, Optional


class ResultsRepository:
    def init_db(self) -> None:
        raise NotImplementedError

    def save(self, filename: str, results: dict) -> None:
        raise NotImplementedError

    def get(self, filename: str) -> Optional[dict]:
        raise NotImplementedError


class InMemoryResultsRepository(ResultsRepository):
    def __init__(self):
        self._rows: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def init_db(self) -> None:
        pass

    def save(self, filename: str, results: dict) -> None:
        with self._lock:
            self._rows[filename] = results

    def get(self, filename: str) -> Optional[dict]:
        with self._lock:
            return self._rows.get(filename)


class PostgresResultsRepository(ResultsRepository):
    def __init__(self, url: str = "postgresql://svar:svar@localhost:5432/svar"):
        import psycopg
        self._conn = psycopg.connect(url, autocommit=True)

    def init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                id BIGSERIAL PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                results JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )

    def save(self, filename: str, results: dict) -> None:
        import json
        self._conn.execute(
            """
            INSERT INTO calls (filename, results) VALUES (%s, %s)
            ON CONFLICT (filename) DO UPDATE SET results = EXCLUDED.results
            """,
            (filename, json.dumps(results)),
        )

    def get(self, filename: str) -> Optional[dict]:
        import json
        row = self._conn.execute(
            "SELECT results FROM calls WHERE filename = %s", (filename,)
        ).fetchone()
        return json.loads(row[0]) if row else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest pipeline/tests/test_results_repo.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/results_repo.py pipeline/tests/test_results_repo.py
git commit -m "feat: add ResultsRepository for PostgreSQL persistence with in-memory twin"
```

---

### Task 5: Runner — run_pipeline orchestration

**Files:**
- Create: `pipeline/runner.py`
- Test: `pipeline/tests/test_runner.py`

**Interfaces:**
- Consumes: `JobContext` + `stage_*` + `free_gpu` from Task 3; `JobStore`/`InMemoryJobStore`/`RedisJobStore` from Task 2; `ResultsRepository`/`InMemoryResultsRepository`/`PostgresResultsRepository` from Task 4; `api.config.SAMPLE_CALLS_DIR`
- Produces — consumed by Tasks 6 and 7:
  - `def run_pipeline(filename: str, job_store: JobStore | None = None, results_repo: ResultsRepository | None = None) -> dict`
    - Defaults: `RedisJobStore()` / `PostgresResultsRepository()` — real infra; tests pass in-memory twins
    - Validates `os.path.join(SAMPLE_CALLS_DIR, filename)` exists → `FileNotFoundError`
    - `job_store.create(filename)`; executes stages in order (below) wrapped in `_timed_stage` (same semantics as dashboard_server.py:159-174, but writing to `job_store.update_stage`); `free_gpu` after `stt`+`acoustic` and after `text_emo` (same points as dashboard_server.py:422, 425)
    - On success: `results = {...}` (exact keys of `api_results`, dashboard_server.py:454-465) → `results_repo.save(filename, results)` → `job_store.finish(filename)`
    - On exception: `job_store.finish(filename, error=str(e))` then re-raise
    - Returns the results dict
  - `def _stage_results(results: dict, keys: list[str]) -> dict` — returns `{k: results.get(k) for k in keys}` (helper for per-stage API endpoints in Task 6)

**Stage order (verbatim from dashboard_server.py:418-430):**
`denoise, diarize, stt, acoustic, [free_gpu], text_emo, [free_gpu], compliance (stage_audit), fusion, qa, crm`

- [ ] **Step 1: Write the failing test**

`pipeline/tests/test_runner.py`:

```python
import pytest

from pipeline.job_store import InMemoryJobStore, STAGES
from pipeline.results_repo import InMemoryResultsRepository
from pipeline import runner, stages


def make_deps():
    return InMemoryJobStore(), InMemoryResultsRepository()


def test_runner_skips_missing_file(tmp_path, monkeypatch):
    job_store, repo = make_deps()
    with pytest.raises(FileNotFoundError):
        runner.run_pipeline("nope.mp3", job_store=job_store, results_repo=repo)


def test_runner_full_flow_with_mocked_stages(tmp_path, monkeypatch):
    job_store, repo = make_deps()
    calls = []
    for sid in [s["id"] for s in STAGES]:
        def make_stage(sid=sid):
            def stage(ctx):
                calls.append(sid)
                ctx.cache[sid] = "done"
            return stage
        monkeypatch.setattr(stages, f"stage_{sid}", make_stage())

    monkeypatch.setattr(stages, "free_gpu", lambda label="": None)

    path = tmp_path / "x.wav"
    path.write_bytes(b"not really audio")
    results = runner.run_pipeline("x.wav", job_store=job_store, results_repo=repo)

    assert calls == [s["id"] for s in STAGES]
    assert job_store.get("x.wav")["status"] == "completed"
    assert repo.get("x.wav") is not None
    assert results["processing_time_s"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pipeline/tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.runner'`

- [ ] **Step 3: Write minimal implementation**

`pipeline/runner.py`:

```python
import os
import time
from typing import List, Optional

from api.config import SAMPLE_CALLS_DIR
from pipeline.job_store import JobStore, RedisJobStore, STAGES
from pipeline.results_repo import PostgresResultsRepository, ResultsRepository
from pipeline import stages
from pipeline.stages import JobContext, free_gpu

STAGE_ORDER = [s["id"] for s in STAGES]


def _timed_stage(filename: str, stage_id: str, job_store: JobStore, fn):
    job_store.update_stage(filename, stage_id, "running")
    t0 = time.time()
    try:
        result = fn()
        elapsed = round(time.time() - t0, 2)
        job_store.update_stage(filename, stage_id, "done", time_s=elapsed)
        return result
    except Exception:
        job_store.update_stage(filename, stage_id, "error")
        raise


def run_pipeline(
    filename: str,
    job_store: Optional[JobStore] = None,
    results_repo: Optional[ResultsRepository] = None,
) -> dict:
    job_store = job_store or RedisJobStore()
    results_repo = results_repo or PostgresResultsRepository()

    filepath = os.path.join(SAMPLE_CALLS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filename} not found")

    ctx = JobContext(filepath=filepath, filename=filename, cache={}, job_store=job_store)
    job_store.create(filename)
    t0 = time.time()
    try:
        _timed_stage(filename, "denoise", job_store, lambda: stages.stage_denoise(ctx))
        _timed_stage(filename, "diarize", job_store, lambda: stages.stage_diarize(ctx))
        _timed_stage(filename, "stt", job_store, lambda: stages.stage_stt(ctx))
        _timed_stage(filename, "acoustic", job_store, lambda: stages.stage_acoustic(ctx))
        free_gpu("stt+acoustic done")

        _timed_stage(filename, "text_emo", job_store, lambda: stages.stage_text_emotion(ctx))
        free_gpu("text_emo done")

        _timed_stage(filename, "compliance", job_store, lambda: stages.stage_audit(ctx))
        _timed_stage(filename, "fusion", job_store, lambda: stages.stage_fusion(ctx))
        _timed_stage(filename, "qa", job_store, lambda: stages.stage_qa(ctx))
        _timed_stage(filename, "crm", job_store, lambda: stages.stage_crm(ctx))

        results = {
            "duration_s": ctx.cache.get("duration"),
            "processing_time_s": round(time.time() - t0, 1),
            "segments": ctx.cache.get("segments"),
            "talk_ratio": ctx.cache.get("talk_ratio", {}),
            "denoise_metrics": ctx.cache.get("denoise_metrics"),
            "role_resolution": ctx.cache.get("role_resolution", {}),
            "fusion": ctx.cache.get("fusion", []),
            "compliance": ctx.cache.get("compliance"),
            "qa": ctx.cache.get("qa"),
            "crm_note": ctx.cache.get("crm_note"),
        }
        results_repo.save(filename, results)
        job_store.finish(filename)
        return results
    except Exception as e:
        job_store.finish(filename, error=str(e))
        raise


def _stage_results(results: dict, keys: List[str]) -> dict:
    return {k: results.get(k) for k in keys}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest pipeline/tests/test_runner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/runner.py pipeline/tests/test_runner.py
git commit -m "feat: add run_pipeline orchestrator with stage timing and result persistence"
```

---

### Task 6: FastAPI app — endpoints, static + audio serving

**Files:**
- Create: `api/schemas.py`
- Create: `api/main.py`
- Test: `api/tests/test_api.py`

**Interfaces:**
- Consumes: `JobStore` (Task 2), `ResultsRepository` (Task 4), `run_pipeline` (Task 5), `api.config` (Task 1)
- Produces — consumed by Task 7 (worker needs nothing from the app) and Task 8 (README):
  - `def create_app(job_store=None, results_repo=None, enqueue=None) -> FastAPI`
    - `job_store`/`results_repo`: injectable; defaults `RedisJobStore(REDIS_URL)` / `PostgresResultsRepository(DATABASE_URL)`
    - `enqueue(filename: str) -> None`: injectable; default enqueues on RQ queue `"svar"` (calls `rq.Queue("svar", connection=redis).enqueue("pipeline.worker.run_pipeline_job", filename)`); if `redis.RedisError` → raise → 503
  - Routes (all responses JSON-matching the legacy server; errors `{"error": str}` with 404/500 like dashboard_server.py:677-684):
    - `GET /health` → `{"status": "ok"}`
    - `GET /api/sample_calls` → `list[str]` (filter `_denoised.wav` like dashboard_server.py:627-633)
    - `POST /api/analyze` body `{"filename": str}` → validate file exists (404 `{"error": "X not found"}`); if progress status is `running` → `{"status": "running", "message": "Pipeline already running"}`; if results exist in repo → `{"status": "completed", "message": "Already analyzed"}`; else `job_store.create` + `enqueue(filename)` + `{"status": "queued", "message": "Pipeline queued"}`
    - `GET /api/progress?file=<filename>` → progress dict or `{"status": "idle", "percent": 0, "stages": {}}` (dashboard_server.py:620-625)
    - `POST /api/results` body `{"filename": str}` → repo results; 404 `{"error": "no results yet"}` if absent
    - `POST /api/denoise|/api/diarize|/api/transcribe|/api/emotion|/api/compliance|/api/qa-score|/api/crm-note` → `_stage_results` slices per legacy handlers (dashboard_server.py:481-558):
      - `denoise`: `["duration_s", "denoise_metrics"]`
      - `diarize`: `["duration_s", "segments", "talk_ratio", "separability", "confidence_method", "role_resolution"]`
      - `transcribe`: `["duration_s", "segments", "talk_ratio", "role_resolution"]`
      - `emotion`: `["duration_s", "segments", "fusion"]`
      - `compliance`: `["compliance", "crm_note"]`
      - `qa-score`: `["qa", "crm_note", "compliance"]`
      - `crm-note`: `["crm_note", "compliance", "qa"]`
    - `GET /audio/{filename}` → `FileResponse` with content-type by extension (dashboard_server.py:635-649); 404 on missing
    - `GET /` → `dashboard/dist/index.html` if exists else `dashboard/index.html`; `GET /favicon.ico` → 204; mount `StaticFiles(directory=dashboard/dist)` at `/` for assets if dist exists (dist files take precedence over SPA route — mount assets via `app.mount("/assets", ...)`; the legacy server also serves arbitrary files under dist, so mount `/` static with `html=True` AFTER registering `/` index route? No — register `/` last in FastAPI route order; mount `StaticFiles` for `/assets` only, plus an explicit `/{path:path}` catch-all serving dist files with the legacy MIME mapping)

- [ ] **Step 1: Write the failing test**

`api/tests/test_api.py`:

```python
import os

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from pipeline.job_store import InMemoryJobStore, STAGES
from pipeline.results_repo import InMemoryResultsRepository


@pytest.fixture
def client():
    job_store = InMemoryJobStore()
    repo = InMemoryResultsRepository()
    enqueued = []

    def fake_enqueue(filename):
        enqueued.append(filename)

    app = create_app(job_store=job_store, results_repo=repo, enqueue=fake_enqueue)
    app.state.enqueued = enqueued
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_sample_calls_returns_list(client):
    res = client.get("/api/sample_calls").json()
    assert isinstance(res, list)
    assert all(isinstance(f, str) for f in res)


def test_analyze_unknown_file_404(client):
    res = client.post("/api/analyze", json={"filename": "nope.mp3"})
    assert res.status_code == 404
    assert "error" in res.json()


def test_analyze_enqueues_and_queues(client, monkeypatch, tmp_path):
    os.makedirs(tmp_path / "data" / "sample_calls", exist_ok=True)
    monkeypatch.setattr("api.main.SAMPLE_CALLS_DIR", str(tmp_path / "data" / "sample_calls"))
    (tmp_path / "data" / "sample_calls" / "a.mp3").write_bytes(b"x")
    res = client.post("/api/analyze", json={"filename": "a.mp3"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "queued"
    assert client.app.state.enqueued == ["a.mp3"]
    assert client.get("/api/progress?file=a.mp3").json()["status"] == "queued"


def test_analyze_completed_if_results_exist(client, monkeypatch, tmp_path):
    os.makedirs(tmp_path / "data" / "sample_calls", exist_ok=True)
    monkeypatch.setattr("api.main.SAMPLE_CALLS_DIR", str(tmp_path / "data" / "sample_calls"))
    (tmp_path / "data" / "sample_calls" / "a.mp3").write_bytes(b"x")
    client.app.state.repo.save("a.mp3", {"qa": {"score": 99}})
    res = client.post("/api/analyze", json={"filename": "a.mp3"}).json()
    assert res["status"] == "completed"
    assert client.app.state.enqueued == []


def test_analyze_running_returns_running(client, monkeypatch, tmp_path):
    os.makedirs(tmp_path / "data" / "sample_calls", exist_ok=True)
    monkeypatch.setattr("api.main.SAMPLE_CALLS_DIR", str(tmp_path / "data" / "sample_calls"))
    (tmp_path / "data" / "sample_calls" / "a.mp3").write_bytes(b"x")
    client.app.state.job_store.create("a.mp3")
    client.app.state.job_store.update_stage("a.mp3", "denoise", "running")
    res = client.post("/api/analyze", json={"filename": "a.mp3"}).json()
    assert res["status"] == "running"


def test_progress_idle_default(client):
    res = client.get("/api/progress?file=zzz.mp3").json()
    assert res["status"] == "idle"
    assert res["percent"] == 0


def test_results_missing_404(client):
    res = client.post("/api/results", json={"filename": "zzz.mp3"})
    assert res.status_code == 404


def test_results_roundtrip(client):
    client.app.state.repo.save("a.mp3", {"qa": {"score": 88}, "crm_note": "n"})
    res = client.post("/api/results", json={"filename": "a.mp3"})
    assert res.json()["qa"]["score"] == 88


def test_stage_endpoints_slice(client):
    client.app.state.repo.save("a.mp3", {"compliance": {"flags": []}, "crm_note": "n", "qa": {"score": 10}})
    assert client.post("/api/compliance", json={"filename": "a.mp3"}).json() == {
        "compliance": {"flags": []}, "crm_note": "n"
    }
    assert client.post("/api/qa-score", json={"filename": "a.mp3"}).json()["qa"]["score"] == 10
    assert client.post("/api/denoise", json={"filename": "a.mp3"}).json() == {
        "duration_s": None, "denoise_metrics": None
    }


def test_audio_serving(client, monkeypatch, tmp_path):
    os.makedirs(tmp_path / "data" / "sample_calls", exist_ok=True)
    monkeypatch.setattr("api.main.SAMPLE_CALLS_DIR", str(tmp_path / "data" / "sample_calls"))
    (tmp_path / "data" / "sample_calls" / "a.mp3").write_bytes(b"audio")
    res = client.get("/audio/a.mp3")
    assert res.status_code == 200
    assert res.content == b"audio"
    assert res.headers["content-type"] == "audio/mpeg"
    assert client.get("/audio/missing.mp3").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest api/tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.main'`

- [ ] **Step 3: Write minimal implementation**

`api/schemas.py`:

```python
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    filename: str
```

`api/main.py`:

```python
import os
import urllib.parse

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from api.config import DASHBOARD_DIR, REDIS_URL, SAMPLE_CALLS_DIR
from api.schemas import AnalyzeRequest
from pipeline.job_store import JobStore, RedisJobStore
from pipeline.results_repo import PostgresResultsRepository, ResultsRepository
from pipeline.runner import _stage_results

STAGE_SLICES = {
    "denoise": ["duration_s", "denoise_metrics"],
    "diarize": ["duration_s", "segments", "talk_ratio", "separability", "confidence_method", "role_resolution"],
    "transcribe": ["duration_s", "segments", "talk_ratio", "role_resolution"],
    "emotion": ["duration_s", "segments", "fusion"],
    "compliance": ["compliance", "crm_note"],
    "qa-score": ["qa", "crm_note", "compliance"],
    "crm-note": ["crm_note", "compliance", "qa"],
}

AUDIO_CONTENT_TYPES = {
    ".mp3": "audio/mpeg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
}


def _default_enqueue(filename: str):
    import redis
    from rq import Queue
    conn = redis.Redis.from_url(REDIS_URL)
    Queue("svar", connection=conn).enqueue("pipeline.worker.run_pipeline_job", filename)


def create_app(
    job_store: JobStore = None,
    results_repo: ResultsRepository = None,
    enqueue=None,
) -> FastAPI:
    job_store = job_store or RedisJobStore(REDIS_URL)
    results_repo = results_repo or PostgresResultsRepository()
    enqueue = enqueue or _default_enqueue

    app = FastAPI(title="SVAR API")
    app.state.job_store = job_store
    app.state.results_repo = results_repo

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/sample_calls")
    def sample_calls():
        return sorted(
            f for f in os.listdir(SAMPLE_CALLS_DIR)
            if f.endswith((".wav", ".mp3", ".opus")) and not f.endswith("_denoised.wav")
        )

    @app.post("/api/analyze")
    def analyze(req: AnalyzeRequest):
        filepath = os.path.join(SAMPLE_CALLS_DIR, req.filename)
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail=f"{req.filename} not found")
        p = job_store.get(req.filename)
        if p and p["status"] == "running":
            return {"status": "running", "message": "Pipeline already running"}
        if results_repo.get(req.filename) is not None:
            return {"status": "completed", "message": "Already analyzed"}
        try:
            job_store.create(req.filename)
            enqueue(req.filename)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"queue unavailable: {e}")
        return {"status": "queued", "message": "Pipeline queued"}

    @app.get("/api/progress")
    def progress(file: str = ""):
        p = job_store.get(file)
        return p or {"status": "idle", "percent": 0, "stages": {}}

    @app.post("/api/results")
    def results(req: AnalyzeRequest):
        res = results_repo.get(req.filename)
        if res is None:
            raise HTTPException(status_code=404, detail="no results yet")
        return res

    for route, keys in STAGE_SLICES.items():

        @app.post(f"/api/{route}")
        def stage_endpoint(req: AnalyzeRequest, _keys: list = keys, _route: str = route):
            res = results_repo.get(req.filename)
            if res is None:
                raise HTTPException(status_code=404, detail="no results yet")
            return _stage_results(res, _keys)

    @app.get("/audio/{filename:path}")
    def audio(filename: str):
        filepath = os.path.join(SAMPLE_CALLS_DIR, urllib.parse.unquote(filename))
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Audio file not found")
        ext = os.path.splitext(filepath)[1].lower()
        return FileResponse(filepath, media_type=AUDIO_CONTENT_TYPES.get(ext, "application/octet-stream"))

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    dist = os.path.join(DASHBOARD_DIR, "dist")
    if os.path.isdir(dist):
        app.mount("/assets", StaticFiles(directory=os.path.join(dist, "assets")), name="assets")

    @app.get("/")
    def index():
        dist_index = os.path.join(dist, "index.html") if os.path.isdir(dist) else None
        fallback = os.path.join(DASHBOARD_DIR, "index.html")
        target = dist_index if dist_index and os.path.exists(dist_index) else fallback
        if not os.path.exists(target):
            raise HTTPException(status_code=404, detail="dashboard not built")
        return FileResponse(target, media_type="text/html")

    return app


app = create_app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest api/tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/schemas.py api/main.py api/tests/test_api.py
git commit -m "feat: add FastAPI app with legacy-compatible API contract, queue, and static serving"
```

---

### Task 7: RQ worker process

**Files:**
- Create: `pipeline/worker.py`
- Test: `pipeline/tests/test_worker.py`

**Interfaces:**
- Consumes: `run_pipeline` (Task 5)
- Produces — consumed by `_default_enqueue` in Task 6 (import path string `"pipeline.worker.run_pipeline_job"`) and Task 8 (README run commands):
  - `def run_pipeline_job(filename: str) -> dict` — thin wrapper calling `run_pipeline(filename)` with real `RedisJobStore` + `PostgresResultsRepository` defaults
  - `def main() -> None` — `Worker(Queue("svar", connection=Redis.from_url(REDIS_URL))).work()`
  - Module entrypoint: `python -m pipeline.worker`

- [ ] **Step 1: Write the failing test**

`pipeline/tests/test_worker.py`:

```python
from pipeline import worker


def test_run_pipeline_job_wrapper_delegates(monkeypatch):
    captured = {}

    def fake_run(filename, **kwargs):
        captured["filename"] = filename
        return {"done": True}

    monkeypatch.setattr(worker, "run_pipeline", fake_run)
    assert worker.run_pipeline_job("a.mp3") == {"done": True}
    assert captured["filename"] == "a.mp3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pipeline/tests/test_worker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.worker'`

- [ ] **Step 3: Write minimal implementation**

`pipeline/worker.py`:

```python
from redis import Redis
from rq import Queue, Worker

from api.config import REDIS_URL
from pipeline.results_repo import PostgresResultsRepository
from pipeline.runner import run_pipeline


def run_pipeline_job(filename: str) -> dict:
    return run_pipeline(filename)


def main() -> None:
    conn = Redis.from_url(REDIS_URL)
    PostgresResultsRepository().init_db()
    worker = Worker(Queue("svar", connection=conn))
    worker.work()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest pipeline/tests/test_worker.py -v`
Expected: PASS

- [ ] **Step 5: Verify RQ can resolve the job function by import path**

Run: `python -c "from rq import get_current_job; import pipeline.worker; print('ok')"`
Expected: prints `ok` (no import errors)

- [ ] **Step 6: Commit**

```bash
git add pipeline/worker.py pipeline/tests/test_worker.py
git commit -m "feat: add RQ worker process for pipeline execution"
```

---

### Task 8: Retire legacy server, update README + Roadmap

**Files:**
- Delete: `dashboard/dashboard_server.py`
- Modify: `README.md`, `Roadmap.md`

**Interfaces:**
- Consumes: everything from Tasks 1-7

- [ ] **Step 1: Update README run instructions**

Replace the "3. Launch Dashboard" and API-endpoint sections with:

```markdown
### 3. Launch (Phase 1 — FastAPI + Redis + Postgres)

Infra first (two containers):
```bash
docker run -d --name svar-redis -p 6379:6379 redis:7
docker run -d --name svar-pg \
  -e POSTGRES_USER=svar -e POSTGRES_PASSWORD=svar -e POSTGRES_DB=svar \
  -p 5432:5432 postgres:16
```

API server:
```bash
python -m uvicorn api.main:app --port 8050
```

Pipeline worker (one per GPU):
```bash
python -m pipeline.worker
```

Open http://localhost:8050, select a sample call, click **Analyze**.
The API enqueues the job and returns immediately; the worker runs
denoise → diarize → STT → emotion → audit and persists results to Postgres.
Progress polling is unchanged (`/api/progress`).
```

Update the API endpoint table: add `GET /health`, note `POST /api/analyze` returns `{status: "queued"|"running"|"completed", message}` and that results persist in Postgres across restarts.

- [ ] **Step 2: Delete the legacy server**

```bash
git rm dashboard/dashboard_server.py
```

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest api/tests pipeline/tests denoising/tests diarization/tests sentiment/tests -q`
Expected: all pass (new tests + existing suites unaffected)

- [ ] **Step 4: Smoke-test the real stack (manual, requires Redis + Postgres running)**

Run:
```bash
python -m uvicorn api.main:app --port 8050 &   # terminal 1
python -m pipeline.worker &                     # terminal 2
curl -s localhost:8050/health
curl -s -X POST localhost:8050/api/analyze -H 'Content-Type: application/json' -d '{"filename": "sample_audio.mp3"}'
curl -s "localhost:8050/api/progress?file=sample_audio.mp3"   # poll until completed
curl -s -X POST localhost:8050/api/results -H 'Content-Type: application/json' -d '{"filename": "sample_audio.mp3"}'
```
Expected: health ok → queued → completed progress → full results JSON (STT/Gemini need the existing credentials; with billing disabled the audit guard applies — results still persist with empty segments)

- [ ] **Step 5: Update Roadmap.md**

In the Summary table change `Phase 1: FastAPI + Redis Queue + Postgres | 🔴 0%` → `✅ Complete`; move the Phase-1 row into the completed grouping; in "Remaining Work" drop item 4 (`Phase 1 — FastAPI backend + Redis job queue + PostgreSQL persistence`) and renumber; add a note under the "Why FastAPI + Redis + Postgres?" architecture note that Phase 1 is implemented (API enqueues → worker executes → Postgres persists, progress via Redis).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: replace http.server dashboard with FastAPI + RQ worker + Postgres persistence"
```

---

## Self-Review

**Spec coverage (Roadmap Phase 1):**
- "Swap to FastAPI — replace dashboard_server.py" → Task 6 + Task 8 (retirement)
- "Native async... Pydantic models" → FastAPI + `api/schemas.py` (Task 6)
- "GET /health" → Task 6
- "Redis queue (RQ)... server only receives file, returns job_id/status immediately" → Task 2 (`RedisJobStore`) + Task 6 (`POST /api/analyze` enqueues, returns `queued` immediately)
- "separate worker process pulls pipeline off queue" → Task 7 (`python -m pipeline.worker`)
- "PostgreSQL keyed by call_id" → Task 4 (`calls` table, `filename` unique key)
- "Do not do all at once" → WebSockets (Phase 2), Docker (Phase 3), Prometheus/tenacity (Phase 4) explicitly NOT included; Phase 2 needs no changes here (progress polling endpoint preserved).
- Roadmap STT cost items (VAD gating, transcript disk cache) — out of scope, still pending; the `JobContext.cache` refactor in Task 3 does not preempt them.

**Placeholder scan:** every task has concrete test code and implementation code or an explicit verbatim-copy rule with exact source line ranges; no TBDs. The Task 3 transformation rule is precise and mechanical (5 enumerated substitutions + line ranges), so no stage body is silently dropped.

**Type consistency:** `JobContext(filepath, filename, cache, job_store)` used identically in Tasks 3 and 5; `JobStore.create/update_stage/finish/get` signatures match across Tasks 2/5/6/7; `ResultsRepository.init_db/save/get` match across Tasks 4/5/6; `run_pipeline(filename, job_store=None, results_repo=None) -> dict` matches between Tasks 5/7; `create_app(job_store, results_repo, enqueue)` consistent between Task 6 test and implementation; stage ids in `STAGE_SLICES` match the legacy API_ROUTES handlers (dashboard_server.py:481-558) and the runner order.

**init_db:** `PostgresResultsRepository().init_db()` (idempotent `CREATE TABLE IF NOT EXISTS`) is called once at worker startup in Task 7 (`worker.main()`), so the table exists before the first job completes.