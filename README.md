# SVAR

**Sentiment and Voice Analytics for Real-time QA** — an end-to-end Hindi call analytics pipeline that denoises recordings, separates speakers, transcribes speech, infers roles, detects emotions, enforces compliance, scores agent quality, and generates CRM notes.

---

## Architecture

```
Raw Audio
  │
  ▼
┌──────────────┐
│  Denoising   │  declip → HPF 80Hz → notch 50Hz → compressor → Wiener
└──────┬───────┘
       ▼
┌──────────────┐
│ Diarization  │  pyannote 3.1 segmentation + SpeechBrain ECAPA-TDNN
└──────┬───────┘
       ▼
┌───────────────────────────────────────┐
│  STT (Google Cloud Chirp 3 V2, hi-IN) │  50s chunks, parallel
└──────┬────────────────────────────────┘
       ▼
┌───────────────────────────────────────┐
│  Role Resolution                      │  Gemini → MuRIL classifier
│  (agent / customer)                   │  → heuristic → first-speaker
└──────┬────────────────────────────────┘
       ▼
┌────────────────────┐   ┌──────────────────────────┐
│  Text Emotion      │   │  Acoustic Emotion        │
│  opus-mt-hi-en     │   │  speaker-normalized      │
│  → DistilRoBERTa   │   │  prosodic profiles       │
└─────────┬──────────┘   └────────────┬─────────────┘
          └──────────┬────────────────┘
                     ▼
          ┌──────────────────┐
          │  Fusion          │  neutral-override, ±conf thresholds
          └─────────┬────────┘
                    ▼
┌───────────────────────────────────────────┐
│  Unified Gemini Audit (ONE LLM call)      │  Compliance violations
│  compliance + QA scorecard + CRM note     │  + QA grade + CRM note
└───────────────────────────────────────────┘
```

The pipeline runs as a job on the Redis queue, executed by a separate RQ worker process (see `pipeline/worker.py`); results persist in Postgres.

---

## Components

| Module | Description |
|---|---|
| `denoising/` | Audio preprocessing: declipping, Butterworth highpass (80 Hz), IIR notch (50 Hz mains hum), dynamic compressor, spectral Wiener denoiser. Benchmarked up to +9.11 dB SNR improvement. |
| `diarization/` | Speaker diarization via `pyannote/speaker-diarization-3.1`, ECAPA-TDNN embeddings via SpeechBrain, false-split detection, silhouette-based confidence, merged-turn splitting, Viterbi re-decoding. |
| `sentiment/stt/` | Google Cloud Speech-to-Text V2 (Chirp 3) with parallel 50s chunk transcription. Requires GCP billing. |
| `sentiment/role_resolver_llm.py` | Gemini-based agent/customer role resolution (primary). |
| `svar/role_inference.py` | MuRIL-based role classifier fallback (heuristic + first-speaker fallback). Disable via `ROLE_INFERENCE_DISABLED=1`. |
| `sentiment/emotion_classifier.py` | Text emotion: local `Helsinki-NLP/opus-mt-hi-en` translation → `j-hartmann/emotion-english-distilroberta-base` |
| `sentiment/acoustic_emotion/` | Rule-based acoustic emotion: prosodic features vs research emotion profiles, speaker-normalized deltas, temperature-scaled confidence |
| `sentiment/fusion_layer.py` | Text+acoustic fusion: neutral-override (acoustic wins when text is neutral), confidence thresholds 0.70/0.65 |
| `sentiment/audit_llm.py` | **Unified Gemini audit** — compliance + QA scorecard + CRM note in ONE call (`gemini-3.5-flash-lite`, 4-model fallback chain, key rotation). Disable via `LLM_AUDIT_DISABLED=1`. |
| `sentiment/compliance_engine.py` | Keyword fast-path + Gemini verification (fallback when unified audit unavailable). |
| `sentiment/qa_scorer.py` | 5-factor weighted QA scoring (fallback). |
| `sentiment/crm_note_generator.py` | TF-IDF extractive CRM summarizer (fallback). |
| `api/main.py` | FastAPI backend (port 8050) — serves the built React app, enqueues pipeline jobs on the Redis queue (RQ), exposes JSON APIs. |
| `pipeline/worker.py` | RQ worker — pulls pipeline jobs off the Redis queue, runs the stages, persists results to Postgres. |
| Redis | RQ job queue + pipeline progress store. |
| PostgreSQL | Persistent per-call results store (survives restarts). |
| `dashboard-ui/` | React 19 + Vite + Tailwind v4 operator dashboard. Builds to `dashboard/dist`. |
| `docs/` | Annotation guidelines. |

---

## Quick Start

### 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
pip install pyannote.audio speechbrain librosa soundfile numpy scikit-learn joblib google-cloud-speech transformers sentencepiece pyyaml
hf auth login   # required: pyannote/speaker-diarization-3.1 is a gated model
```

### 2. Credentials

Place service account key at `credentials/gcloud-stt.json` (project `sunohq`, requires billing enabled for Chirp 3 V2).

Place Gemini API keys at `credentials/gemini-api-keys.json`:
```json
{"keys": ["key1", "key2", ...]}
```

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

### 4. Frontend Development (optional)

```bash
cd dashboard-ui
npm install
npm run dev        # http://localhost:5174 (5173 may be taken; proxy → :8050)
npm run build      # rebuilds dashboard/dist served by the Python server
```

### 5. Run Individual Stages

```bash
python -m denoising.pipeline      # Denoising
python -m diarization.pipeline    # Diarization
python -m sentiment.pipeline      # Full sentiment pipeline
```

---

## API Endpoints (port 8050)

| Endpoint | Description |
|---|---|
| `GET /` | Dashboard (built React app) |
| `GET /health` | Liveness check — returns `{"status": "ok"}` if the API is up |
| `GET /api/sample_calls` | List available audio files |
| `GET /api/progress?file=X` | Pipeline progress polling |
| `POST /api/analyze` | Enqueue pipeline job (denoise → diarize → stt → acoustic → text_emo → compliance → fusion → qa → crm); returns immediately with `{status: "queued"\|"running"\|"completed", message}` |
| `POST /api/results` | Complete per-segment results |
| `POST /api/denoise` `/api/diarize` `/api/transcribe` `/api/emotion` `/api/compliance` `/api/qa-score` `/api/crm-note` | Cached per-stage results |
| `GET /audio/<file>` | Serve raw audio files |

Jobs are executed by a separate RQ worker process, and results persist in Postgres across restarts.

---

## Environment Variables

| Variable | Effect |
|---|---|
| `SVAR_PORT` | API server port (default `8050`) |
| `SVAR_REDIS_URL` | Redis connection URL for the RQ queue and progress store (default `redis://localhost:6379/0`) |
| `SVAR_DATABASE_URL` | PostgreSQL connection URL for results persistence (default `postgresql://svar:svar@localhost:5432/svar`) |
| `LLM_AUDIT_DISABLED=1` | Skip unified Gemini audit (falls back to local compliance/QA/CRM) |
| `LLM_COMPLIANCE_DISABLED=1` | Skip Gemini in compliance engine |
| `LLM_CRM_DISABLED=1` | Skip Gemini CRM note generation |
| `ROLE_INFERENCE_DISABLED=1` | Skip role inference (no agent/customer mapping) |
| `STT_VAD_DISABLED=1` | Send full audio to STT instead of VAD-gated speech chunks |

---

## GPU Notes

- **RTX 2050 (4GB VRAM)**: pyannote runs on GPU, then freed before loading translation/emotion models (`_free_gpu` between stages). STT and the audit LLM are cloud-based.
- `torchcodec` is not supported (missing `libnvrtc.so.13`) — the warning is benign; audio loading uses `soundfile` + `librosa`.

---

## Project Structure

```
SVAR/
├── credentials/          GCloud STT + Gemini keys (.gitignored)
├── dashboard/            Python dashboard server + built React app (dist/)
├── dashboard-ui/         React + Vite + Tailwind frontend source
├── data/sample_calls/    Sample audio files
├── denoising/            Audio preprocessing pipeline
├── diarization/          Speaker diarization
├── docs/                 Annotation guidelines
├── sentiment/            STT, roles, emotion, acoustic, audit
└── svar/                 MuRIL role classifier fallback
```

---

## License

MIT
