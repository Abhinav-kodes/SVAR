# Docker Compose Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `docker compose up -d` launches the entire SVAR stack — FastAPI API (port 8050), RQ worker on the host GPU, Redis, Postgres — in one command.

**Architecture:** Single multi-stage `Dockerfile` (node:22 stage builds `dashboard/dist`, python:3.11-slim stage installs `requirements.txt` with cu128 torch wheels and copies the repo). `compose.yml` defines 4 services: `redis:7` + `postgres:16` (internal-only, healthchecked) and `api` + `worker` sharing the one built image with different commands. Credentials are bind-mounted read-only; HF models download to a named volume at first worker boot.

**Tech Stack:** Docker 29 / Compose v5, Dockerfile multi-stage, npm ci (dashboard-ui), python:3.11-slim + pip, cu128 torch wheels, NVIDIA GPU.

## Global Constraints

- No changes to application code, env-var names, or test files — Docker files only plus docs.
- Credentials (`credentials/gcloud-stt.json`, `credentials/gemini-api-keys.json`) are NEVER baked into the image: no `COPY` of `credentials/` in the Dockerfile, read-only bind mount at runtime.
- `HF_TOKEN` comes from the host `.env` file only (`${HF_TOKEN:-}`).
- Redis and Postgres publish NO host ports — internal compose network only.
- The GPU worker matches local dev: cu128 wheels, `gpus: all`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `CUDA_MODULE_LOADING=LAZY` (small-VRAM mitigation — host GPU is RTX 2050 ~3.5 GB).
- `data/` is bind-mounted at runtime (`./data:/app/data`), so image content under `/app/data` is shadowed; `data/transcripts/` is excluded from the build context (large cache).
- Tests run with `venv/bin/python -m pytest <file> -q` from the repo root.

---

### Task 1: `Dockerfile` + `.dockerignore`

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

**Interfaces:**
- Produces: multi-stage image where stage 2 has `WORKDIR /app`, `PYTHONUNBUFFERED=1`, the repo at `/app`, and `dashboard/dist` containing the freshly built React app. Used by both `api` and `worker` services in Task 2.

- [ ] **Step 1: Write the `Dockerfile`**

```dockerfile
# Stage 1: build the React dashboard
FROM node:22 AS dashboard-build
WORKDIR /app/dashboard-ui
COPY dashboard-ui/package.json dashboard-ui/package-lock.json ./
RUN npm ci
COPY dashboard-ui/ ./
RUN npm run build

# Stage 2: Python runtime (shared by api + worker services)
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY --from=dashboard-build /app/dashboard/dist ./dashboard/dist
```

- [ ] **Step 2: Write the `.dockerignore`**

```
venv/
.venv/
.git/
node_modules/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.env
credentials/
dashboard/dist/
data/transcripts/
```

- [ ] **Step 3: Validate the Dockerfile syntax**

Run: `docker build --check -f Dockerfile . 2>&1 | tail -5`
Expected: exit 0, no syntax errors (BuildKit prints a summary; a `docker build --check` pass means the Dockerfile parses).

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat: add multi-stage Dockerfile and .dockerignore"
```

---

### Task 2: `compose.yml` + `.env.example`

**Files:**
- Create: `compose.yml`
- Create: `.env.example`

**Interfaces:**
- Consumes: the Task 1 image (services reference `build: .`).
- Produces: `docker compose config` resolves 4 services (redis, postgres, api, worker); `api` on host port 8050; env defaults matching `api/config.py` (SVAR_REDIS_URL, SVAR_DATABASE_URL).

- [ ] **Step 1: Write `compose.yml`**

```yaml
services:
  redis:
    image: redis:7
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 10
    restart: unless-stopped

  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: svar
      POSTGRES_PASSWORD: svar
      POSTGRES_DB: svar
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U svar"]
      interval: 5s
      timeout: 3s
      retries: 10
    restart: unless-stopped

  api:
    build: .
    command: uvicorn api.main:app --host 0.0.0.0 --port 8050
    ports:
      - "8050:8050"
    environment:
      SVAR_REDIS_URL: redis://redis:6379/0
      SVAR_DATABASE_URL: postgresql://svar:svar@postgres:5432/svar
    volumes:
      - ./credentials:/app/credentials:ro
      - ./data:/app/data:ro
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    restart: unless-stopped

  worker:
    build: .
    command: python -m pipeline.worker
    gpus: all
    environment:
      SVAR_REDIS_URL: redis://redis:6379/0
      SVAR_DATABASE_URL: postgresql://svar:svar@postgres:5432/svar
      HF_TOKEN: ${HF_TOKEN:-}
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      CUDA_MODULE_LOADING: LAZY
    volumes:
      - ./credentials:/app/credentials:ro
      - ./data:/app/data
      - hf_cache:/root/.cache/huggingface
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    restart: unless-stopped

volumes:
  pgdata:
  hf_cache:
```

- [ ] **Step 2: Write `.env.example`**

```
HF_TOKEN=hf_xxx              # required once for gated pyannote model
# optional overrides:
# SVAR_REDIS_URL=redis://redis:6379/0
# SVAR_DATABASE_URL=postgresql://svar:svar@postgres:5432/svar
```

- [ ] **Step 3: Validate the compose file**

Run: `docker compose config 2>&1 | tail -20`
Expected: resolved config prints (services redis, postgres, api, worker; api exposes `8050:8050`; worker has `deploy.resources.reservations.devices` with GPU count 1).

Also run: `docker compose config --services`
Expected: `redis` `postgres` `api` `worker` (one per line).

- [ ] **Step 4: Commit**

```bash
git add compose.yml .env.example
git commit -m "feat: add docker compose stack with api, worker, redis, postgres"
```

---

### Task 3: Full build + run verification (long-running)

**Files:**
- None new (verify Task 1-2 artifacts end-to-end; commit fixes if any).

**Interfaces:**
- Consumes: `compose.yml` + `Dockerfile` from Tasks 1-2; host `./credentials` (must exist — it does), `./data` (exists), `.env` with `HF_TOKEN` (Step 1).

- [ ] **Step 1: Create the local env file**

Run: `cp .env.example .env`
Then ask the user to put their real `HF_TOKEN` in `.env`. If the user declines or has no token, record the deviation and expect the analyze step (Step 6) to fail at the diarize stage (pyannote gated model) — everything else still verifies.

- [ ] **Step 2: Build the image**

Run: `docker compose build 2>&1 | tail -5`
Expected: builds with no errors. This is the long step (~15-30 min: cu128 torch wheels download ~3-4 GB; node stage installs dashboard deps). Timeout for the command: 3600000 ms (1 hour).

- [ ] **Step 3: Start the stack**

Run: `docker compose up -d && docker compose ps`
Expected: 4 services; redis + postgres `(healthy)`; api + worker running.

- [ ] **Step 4: Verify the API**

Run: `curl -s localhost:8050/health`
Expected: `{"status":"ok"}`

Run: `curl -s localhost:8050/api/sample_calls`
Expected: JSON list of audio files (from the `./data/sample_calls` bind mount).

- [ ] **Step 5: Verify the dashboard**

Run: `curl -s localhost:8050/ | head -5`
Expected: HTML referencing the built asset hashes (image-built `dashboard/dist`, not a stale host copy).

- [ ] **Step 6: Analyze one sample call end-to-end**

Run:
```bash
curl -s -X POST localhost:8050/api/analyze -H 'Content-Type: application/json' -d '{"filename":"<first file from /api/sample_calls>"}'
```
Expected: `{"status":"queued",...}`. Then poll:

```bash
for i in $(seq 1 60); do
  curl -s "localhost:8050/api/progress?file=<same file>" | python3 -c "import sys,json; p=json.load(sys.stdin); print(p['status'], p.get('current_stage',''))"
  sleep 5
done
```
Expected: transitions `queued → running (denoise → … → crm) → completed`, then:
- `curl -s -X POST localhost:8050/api/results -H 'Content-Type: application/json' -d '{"filename":"<file>"}'` returns the full analysis.
- `ls data/transcripts/` shows a `<sha1>-hi.json` cache file (worker's writable `./data` mount wrote through).
- Worker logs show no GPU errors: `docker compose logs worker 2>&1 | grep -iE "error|traceback" | head -5` → empty.

Notes: if `HF_TOKEN` is missing, this step fails at diarize with a clear huggingface error — record it as an environment limitation, not a code bug. The `/ws/progress` live path is already covered by the api/tests/test_ws_progress.py suite; live ws verification is optional here (use `websocat` or the browser devtools if available).

- [ ] **Step 7: Tear down**

Run: `docker compose down`
Expected: containers stop, named volumes remain. (`docker compose down -v` would also drop `pgdata` + `hf_cache` — mention but do NOT run.)

- [ ] **Step 8: Commit any fixes**

If any step required a fix to the Dockerfile/compose files: commit as `fix: <description>`. If everything passed without changes, no commit (record the verification results instead).

---

### Task 4: Docs

**Files:**
- Modify: `Roadmap.md` (Phase 3 section ~line 210, summary table ~line 240, Remaining Work ~line 248)
- Modify: `README.md` (Quick Start sections ~lines 76-115, env table ~line 161)

**Interfaces:**
- Consumes: nothing new.

- [ ] **Step 1: Update `Roadmap.md`**

- Phase 3 section header: change `### Phase 3 — Containerization & Deployment` to `### Phase 3 — Containerization & Deployment ✅ COMPLETE`.
- Summary table row: change `| Phase 3: Docker Compose | 🔴 0% | API + worker Dockerfiles, compose stack |` to `| Phase 3: Docker Compose | 🟢 100% | one image, api+worker services, compose stack |`.
- Remaining Work item 5: change `5. **Phase 3** — Docker Compose deployment` to `5. ~~Phase 3~~ ✅ **Complete** — `docker compose up -d` runs API + GPU worker + Redis + Postgres (single multi-stage image, credentials bind-mounted, HF models in a named volume)`.
- Remaining Work item 6 stays as Phase 4 (no renumber needed — it is already last).

- [ ] **Step 2: Update `README.md`**

- Insert a new `### 3. Docker (recommended)` section directly above the current `### 3. Launch (Phase 1 — FastAPI + Redis + Postgres)` section:

```markdown
### 3. Docker (recommended)

One command runs the whole stack (API + GPU worker + Redis + Postgres):

```bash
cp .env.example .env   # add your HF_TOKEN (gated pyannote model)
docker compose up -d   # first build downloads ~4 GB of cu128 torch wheels
```

Open http://localhost:8050 and click **Analyze**. Models download to a
named volume on first worker boot; credentials stay on the host
(`./credentials` is mounted read-only). Stop with `docker compose down`
(add `-v` to also drop the postgres + models volumes).
```

- Renumber the following sections: `### 3. Launch (Phase 1 — FastAPI + Redis + Postgres)` → `### 4. Launch (Manual — FastAPI + Redis + Postgres)`, `### 4. Frontend Development (optional)` → `### 5. Frontend Development (optional)`, `### 5. Run Individual Stages` → `### 6. Run Individual Stages`.
- Env table: add row `| `HF_TOKEN` | Hugging Face token for the gated pyannote diarization model (worker only; not needed for the API) |`.

- [ ] **Step 3: Run the full test suite**

Run: `venv/bin/python -m pytest -q`
Expected: **91 passed, 3 skipped, 4 failed** — unchanged from the Phase 2 baseline (no application code touched; the 4 failures are the pre-existing, unrelated ones: `diarization/tests/test_pipeline.py::TestDiarizationPipeline::test_diarization_pipeline_execution` and 3× `sentiment/tests/test_stt_transcriber.py` method-drift tests).

- [ ] **Step 4: Commit**

```bash
git add Roadmap.md README.md
git commit -m "docs: mark Phase 3 docker compose complete"
```