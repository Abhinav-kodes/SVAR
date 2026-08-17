# Docker Compose Stack Design Spec

**Date:** 2026-08-18
**Status:** Approved

## Problem

Running SVAR locally requires manual infra (docker run Redis + Postgres),
a GPU-enabled Python env, credentials, and a separate worker process. The
goal: `docker compose up` spins up the whole stack (API + worker + Redis +
Postgres) in one command, on the developer's own GPU machine.

## Goals

- One command (`docker compose up -d`) launches: FastAPI API (port 8050),
  RQ pipeline worker, Redis, Postgres.
- Worker uses the host GPU (NVIDIA RTX 2050, ~3.5 GB VRAM) exactly like the
  local dev setup (cu128 torch wheels).
- Credentials (`credentials/gcloud-stt.json`, `credentials/gemini-api-keys.json`)
  stay on the host — bind-mounted read-only, never baked into the image.
- ML models (pyannote 3.1, ECAPA, opus-mt, emotion BERT) download at first
  worker boot into a named volume; image stays lean of model weights.
- All existing behavior preserved: `/api/progress`, `/ws/progress`, disk
  transcript cache (`data/transcripts/`), dashboard served from
  `dashboard/dist`.

## Non-Goals

- No CI, no registry push, no multi-arch, no GPU-less fallback profile.
- No changes to application code paths or env-var names already in use.

## Architecture

```
compose.yml
├── redis        redis:7            (healthcheck: redis-cli ping)
├── postgres     postgres:16        (svar/svar/svar, volume pgdata)
├── api          built image        (uvicorn api.main:app, :8050)
│                  mounts: ./credentials:/app/credentials:ro
│                          ./data:/app/data:ro
└── worker       built image        (python -m pipeline.worker, gpus: all)
                   mounts: ./credentials:/app/credentials:ro
                           ./data:/app/data          (rw — transcripts cache)
                           hf_cache:/root/.cache/huggingface
```

Single multi-stage `Dockerfile`:

1. `node:22` stage — `npm ci && npm run build` in `dashboard-ui/`
   → produces `dashboard/dist`.
2. `python:3.11-slim` stage — `pip install -r requirements.txt`
   (extra-index-url cu128 torch wheels), copy repo including built dist.

Both `api` and `worker` services use this one image with different
`command:` entries; the ~7 GB torch layer is shared.

## Components

### Dockerfile

- Multi-stage as above. `WORKDIR /app`, repo copied to `/app`.
- No `COPY` of `credentials/` — mounted at runtime.
- `ENV PYTHONUNBUFFERED=1`.

### compose.yml

Services (details):

- `redis`: image `redis:7`, `healthcheck: ["CMD", "redis-cli", "ping"]`,
  no published ports (internal only), `restart: unless-stopped`.
- `postgres`: image `postgres:16`, env `POSTGRES_USER=svar`,
  `POSTGRES_PASSWORD=svar`, `POSTGRES_DB=svar`, volume `pgdata:/var/lib/postgresql/data`,
  healthcheck `pg_isready -U svar`, no published ports (internal only),
  `restart: unless-stopped`.
- `api`: `build: .`, command `uvicorn api.main:app --host 0.0.0.0 --port 8050`,
  `ports: ["8050:8050"]`, env `SVAR_REDIS_URL=redis://redis:6379/0`,
  `SVAR_DATABASE_URL=postgresql://svar:svar@postgres:5432/svar`,
  `depends_on: {redis: {condition: service_healthy}, postgres: {condition: service_healthy}}`,
  volumes `./credentials:/app/credentials:ro`, `./data:/app/data:ro`,
  `restart: unless-stopped`.
- `worker`: same image, command `python -m pipeline.worker`,
  `gpus: all`, env as api plus `HF_TOKEN=${HF_TOKEN:-}`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  `CUDA_MODULE_LOADING=LAZY`, volumes `./credentials:/app/credentials:ro`,
  `./data:/app/data`, `hf_cache:/root/.cache/huggingface`,
  `depends_on` healthy redis+postgres, `restart: unless-stopped`.

### .env.example

```
HF_TOKEN=hf_xxx              # required once for gated pyannote model
# optional overrides:
# SVAR_REDIS_URL=redis://redis:6379/0
# SVAR_DATABASE_URL=postgresql://svar:svar@postgres:5432/svar
```

### .dockerignore

`venv/`, `data/transcripts/`, `.git/`, `node_modules/`, `__pycache__/`,
`*.pyc`, `.env`.

## Error Handling & Operations

- Worker restarts (`unless-stopped`) if it crashes; RQ job survives in Redis.
- If `HF_TOKEN` is unset, pyannote download fails on first boot — worker
  logs a clear error; setting the env + restarting fixes it.
- If credentials are missing from `./credentials`, the API/worker start but
  STT/Gemini calls fail at stage time (same as local dev today).
- Transcripts cache dir is created by the worker (writable `./data` mount).

## Security

- Secrets never baked into the image: `credentials/` is a read-only
  bind mount, `HF_TOKEN` only in container env (from host `.env`).
- No ports exposed for Redis/Postgres — internal network only.
- `.dockerignore` prevents `credentials/`-adjacent leakage (explicitly
  excludes venv and git).

## Testing / Verification

1. `docker compose config` — validate the compose file.
2. `docker compose build` — image builds (torch layer ~3-4 GB download).
3. `docker compose up -d` — all containers healthy.
4. `curl localhost:8050/health` → `{"status": "ok"}`.
5. `curl localhost:8050/api/sample_calls` → lists files from `./data/sample_calls`.
6. Analyze one sample call; confirm progress over `/ws/progress` and
   `data/transcripts/` cache file appears (worker write path).
7. `docker compose down` cleanly stops; `docker compose down -v` also drops
   pgdata + hf_cache volumes.

## Docs

- Roadmap.md: Phase 3 → ✅ COMPLETE, summary table row
  `| Phase 3: Docker Compose | 🟢 100% | one image, api+worker services, compose stack |`,
  Remaining Work item 5 ✅ → renumber Phase 4 → 6.
- README.md: new "Docker (recommended)" launch section above the manual
  one; env table gains `HF_TOKEN` row.