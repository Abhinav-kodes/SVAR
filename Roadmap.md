# SVAR — Call Analytics Pipeline Roadmap
*Updated to reflect current codebase state — 17 Aug 2026*

---

## Project Architecture (Current)

```
RAW CALL AUDIO
      ↓
┌─────────────────────────────────┐
│  PART 1: DENOISING      ✅ DONE │  DSP chain: declip → Butterworth HPF
│  + ENHANCEMENT PIPELINE         │  → IIR notch → compressor → Wiener
└──────────┬──────────────────────┘
           ↓ clean audio
┌─────────────────────────────────┐
│  PART 2: DIARIZATION    ✅ DONE │  pyannote/speaker-diarization-3.1
│  + CONFIDENCE SCORING           │  + SpeechBrain ECAPA-TDNN embeddings
└──────┬──────────┬───────────────┘  + false-split detection, Viterbi
       ↓          ↓
  Agent        Customer
  segments     segments
       ↓
┌─────────────────────────────────┐
│  PART 3: STT + EMOTION  ✅ DONE │  Chirp 3 V2 STT → Gemini role resolver
│  + ROLE INFERENCE               │  → translate+DistilRoBERTa text emotion
└──────┬──────────────────────────┘  + speaker-normalized acoustic emotion
       ↓                          → gated fusion (neutral-override)
   TurnPrediction per segment
   (emotion, sentiment, confidence)
       ↓
┌─────────────────────────────────┐
│  PART 4: UNIFIED AUDIT  ✅ DONE │  ONE Gemini Flash call: compliance
│  Compliance + QA + CRM          │  violations + 5-factor QA scorecard
└──────────┬──────────────────────┘  + CRM note. Local fallbacks retained.
           ↓
┌─────────────────────────────────┐
│  DASHBOARD + INTEGRATION ✅ 90% │  React operator UI + Python server,
│  Sequential background pipeline │  progress polling, per-stage APIs
└─────────────────────────────────┘
```

---

## Part 1 — Denoising + Enhancement Pipeline ✅ COMPLETE

All components implemented, tested, and benchmarked. Benchmarked on 3 sample calls with SNR improvements up to +9.11 dB.

- `denoising/audio_loader.py` — load WAV/MP3/OPUS, stereo→mono, resample to 16kHz via librosa
- `denoising/snr_calculator.py` — hybrid VAD + spectral-percentile SNR (not naive FFT)
- `denoising/clipping_detector.py` — consecutive-run clipping detection (≥3 flatline samples)
- `denoising/vad_basic.py` — RMS energy-threshold VAD with 10th-percentile noise floor
- `denoising/silence_ratio.py` — RMS-based silence ratio via VAD
- `denoising/highpass_filter.py` — 5th-order Butterworth high-pass at 80 Hz
- `denoising/notch_filter.py` — IIR notch at 50 Hz (Indian mains hum), Q=30
- `denoising/compressor.py` — RMS-based dynamic range compressor (-20 dBFS, 3:1 ratio)
- `denoising/declipper.py` — cubic spline interpolation for clipped peaks (≥0.99 threshold)
- `denoising/spectral_denoiser.py` — STFT Wiener filter (25ms Hamming, 50% overlap)
- `denoising/enhancement_pipeline.py` — chains: declip → HPF → notch → compressor
- `denoising/pipeline.py` — `DenoiserPipeline` class returning SNR before/after, clipping/silence ratios, hum removal, audio quality grade
- Tests: `denoising/tests/test_synthetic.py`, `test.py` (batch QA → `quality_report.json`), `benchmark.py` (CSV export)

**Benchmark Results:**
| File | Input SNR | Output SNR | Improvement | Grade |
|---|---|---|---|---|
| sample_audio.mp3 | 54.09 dB | 53.91 dB | -0.18 dB | PASS |
| sample_audio_2.opus | 38.02 dB | 47.13 dB | **+9.11 dB** | PASS |
| sample_audio_3.opus | 33.97 dB | 38.71 dB | **+4.73 dB** | PASS |

---

## Part 2 — Speaker Diarization + Confidence ✅ COMPLETE

> **Architecture:** pyannote/speaker-diarization-3.1 for segmentation + SpeechBrain ECAPA-TDNN embeddings + sklearn silhouette scoring for confidence.

- `diarization/pipeline.py` — `DiarizationPipeline` class
  - `pyannote/speaker-diarization-3.1` (HuggingFace, lazy-loaded)
  - Hook-based chunk embeddings + segmentation data + global centroids
  - Talk ratio computation (agent/customer/overlap durations)
  - Configurable `num_speakers`, `min_speakers`, `max_speakers`
- `diarization/speaker_embedder.py` — SpeechBrain ECAPA-TDNN module
  - `extract_segment_embeddings()` — batch embedding extraction
  - `split_merged_turns()` — detect and split merged turns
  - `reassign_speakers()` — anchor-based reclassification
  - `decode_region()` — Viterbi sequence decoder with transition priors
- `diarization/change_detector.py` — ML-based false split detection + `detect_single_speaker()`
- `diarization/confidence.py` — silhouette-based segment confidence + rolling separability

---

## Part 3 — STT + Emotion Analysis + Role Inference ✅ COMPLETE

### STT (Google Cloud Chirp 3 V2)
- `sentiment/stt/stt_transcriber.py` — Speech-to-Text V2 with Chirp 3 model (`hi-IN`), parallel 50s chunk transcription (5 workers), word-level timestamps mapped back onto diarized segments
- ⚠️ **Requires GCP billing** on project `sunohq` — 403 without it (and the downstream audit must not run on empty transcripts)

### Role Inference (Gemini-first)
- `sentiment/role_resolver_llm.py` — Gemini-based agent/customer mapping (primary), key rotation
- `svar/role_inference.py` + `svar/models/role_classifier.py` — MuRIL classifier fallback chain: classifier → Hindi self-introduction heuristic (first 12 turns) → first-speaker=agent
- `ROLE_INFERENCE_DISABLED=1` env var to skip

### Text Emotion
- `sentiment/emotion_classifier.py` — local `Helsinki-NLP/opus-mt-hi-en` translation → `j-hartmann/emotion-english-distilroberta-base` classification (GPU)

### Acoustic Emotion
- `sentiment/acoustic_emotion/` — 15 prosodic features per segment (pitch, RMS, jitter, shimmer, ZCR, pause ratio…), speaker-normalized deltas, weighted-distance against research emotion profiles, temperature scaling

### Fusion
- `sentiment/fusion_layer.py` — text+acoustic fusion with **neutral-override** (acoustic wins when text=neutral but acoustic detects non-neutral), confidence thresholds (text 0.70 / acoustic 0.65)

---

## Part 4 — Unified Audit: Compliance + QA + CRM ✅ COMPLETE

> **Architecture:** Compliance, QA scorecard, and CRM note are generated in **ONE Gemini Flash call** (`sentiment/audit_llm.py`), saving quota and guaranteeing analytical consistency.

- `sentiment/audit_llm.py` — `GeminiAuditEngine`
  - Single prompt → `{compliance, qa, crm_note}` JSON
  - Compliance: RBI/IRDAI violations, abusive language, threats (per-segment flags)
  - QA: 0–100 score, grade A–F, 5 weighted components
  - CRM: executive summary, key points, recommended action
  - Model: `gemini-3.5-flash-lite` with fallback chain (`2.5-flash` → `3.6-flash` → `3.1-flash-lite` → `3.5-flash`)
  - Round-robin key rotation from `credentials/gemini-api-keys.json`
  - `LLM_AUDIT_DISABLED=1` to skip
- Local fallbacks (used when audit unavailable/fails):
  - `sentiment/compliance_engine.py` + `compliance_llm.py` — keyword fast-path (30+ Hindi abuse words, RBI/IRDAI regex, Levenshtein fuzzy match) + Gemini verification
  - `sentiment/qa_scorer.py` — 5-factor weighted scoring (customer_sentiment 0.30, compliance 0.25, agent_stability 0.20, intent_resolution 0.15, talk_ratio 0.10), grade A(85+) B(70+) C(55+) D(<55)
  - `sentiment/crm_note_generator.py` — TF-IDF extractive summarizer

---

## Dashboard + Integration ✅ 90% COMPLETE

### React Operator UI (`dashboard-ui/`)
- React 19 + Vite 8 + Tailwind CSS v4, framer-motion, recharts, lucide-react
- Task-first operator workflow: Summary, Denoising, Diarization, Transcript, Emotion, Compliance, QA, CRM views
- Builds to `dashboard/dist` (served by the Python server); dev mode on :5173 with `/api` + `/audio` proxy to :8050

### Python Dashboard Server (`dashboard/dashboard_server.py`, port 8050)
- Serves built React app (fallback `dashboard/index.html`)
- `POST /api/analyze` — sequential background pipeline (daemon thread + lock)
- `GET /api/progress?file=X` — stage-by-stage progress polling
- Stage order: `denoise → diarize → stt → acoustic → text_emo → audit → fusion`
- GPU freed between stages (`_free_gpu` — models unloaded + `torch.cuda.empty_cache`)
- Individual cached endpoints: `/api/denoise`, `/api/diarize`, `/api/transcribe`, `/api/emotion`, `/api/compliance`, `/api/qa-score`, `/api/crm-note`, `/api/results`

---

## Integration + Polish 🔴 NOT STARTED

### 🔴 FastAPI Backend
- Replace `http.server` with async FastAPI: JWT auth (HS256), Pydantic models, `/health`

### 🔴 Celery + Task Queue
- Celery + RabbitMQ for async pipeline execution (task chain: denoise → diarize → analyze)

### 🔴 Database + Caching
- MongoDB schemas (Call, Segment, Agent); Redis cache for dashboard queries (TTL 5 min)

### 🔴 Docker Compose
- `Dockerfile` for API + worker; `docker-compose.yml`: api, worker, rabbitmq, mongodb, redis

---

## Summary

| Part | Status | Key Components |
|---|---|---|
| Denoising + Enhancement | ✅ Complete | declip, Butterworth HPF, IIR notch, compressor, Wiener denoiser |
| Diarization | ✅ Complete | pyannote 3.1, SpeechBrain ECAPA-TDNN, change detector, confidence |
| STT + Emotion + Roles | ✅ Complete | Chirp 3 V2, Gemini role resolver, translate+DistilRoBERTa, acoustic profiles, fusion |
| Unified Audit (Compliance + QA + CRM) | ✅ Complete | single Gemini call + local fallbacks |
| Dashboard + Integration | ✅ 90% | React operator UI, sequential background pipeline, progress polling |
| Integration + Polish | 🔴 0% | FastAPI, Celery, MongoDB, Docker Compose |

### Remaining Work (priority order)
1. **Guard against hallucinated audits** — skip/flag the unified audit when the STT transcript is empty (currently Gemini confabulates results on empty transcripts)
2. **FastAPI Backend** — replace `http.server` with proper async API
3. **Docker Compose** — containerized deployment
4. **Celery + task queue** — async pipeline execution
5. **MongoDB + Redis** — persistence and caching

---

## Architecture Notes

### Why Translate + DistilRoBERTa instead of a trained Hindi model?
The original plan (Contextual MuRIL, WavLM, trained fusion) was replaced with a zero-training approach: local `opus-mt-hi-en` translation followed by `emotion-english-distilroberta-base` classification. Fast, GPU-local, no training data/checkpoints required.

### Why Gemini-first role resolution?
A Gemini call resolves agent/customer mapping with full conversational context; MuRIL classifier + heuristics are retained as offline fallbacks (and `ROLE_INFERENCE_DISABLED=1` can force the heuristic path).

### Why a Unified Audit?
Compliance, QA, and CRM note share the same transcript evidence. One Gemini call (66% fewer LLM invocations) keeps scores, flags, and narrative mutually consistent.

### Why sequential pipeline stages?
Stages were made sequential (was: ThreadPoolExecutor) to respect the 4GB GPU — models are unloaded between stages via `_free_gpu`, avoiding CUDA OOM.

---

## Known Issues

1. **STT requires GCP billing** — Chirp 3 V2 returns 403 `BILLING_DISABLED` without billing on the `sunohq` project; transcripts come back empty.
2. **Empty-transcript hallucination** — with no STT text, the Gemini audit currently fabricates compliance flags/QA scores from segment timestamps alone. Guard is the #1 remaining fix.
3. **torchcodec warning** — `libnvrtc.so.13` missing; benign (audio uses soundfile/librosa).
4. **Role classifier is untrained** — MuRIL role classifier has no checkpoint; Gemini resolver is the primary path.

---

## Dataset Citations

```
EmoInHindi (LREC 2022) — IIT Patna          (preprocessing/training stack removed Aug 2026)
Kaggle Hindi Call Center Audio — @infobayai — 760+ hours dual-channel Hindi telephony (sample calls)
Vaani Corpus — IISc Bangalore               — 150,000+ hours, 22+ Indian languages
```