# SVAR — Call Analytics Pipeline Roadmap
*Updated to reflect current codebase state — 26 Jul 2026*

---

## Project Architecture (Current)

```
RAW CALL AUDIO
      ↓
┌─────────────────────────────────┐
│  PART 1: DENOISING      ✅ DONE │  Built from scratch: DSP chain
│  + ENHANCEMENT PIPELINE         │  Butterworth HPF, IIR notch,
└──────────┬──────────────────────┘  compressor, declipper, Wiener
           ↓ clean audio
┌─────────────────────────────────┐
│  PART 2: DIARIZATION    ✅ DONE │  pyannote/speaker-diarization-3.1
│  + ROLE INFERENCE               │  + SpeechBrain ECAPA-TDNN embeddings
└──────┬──────────┬───────────────┘  + MuRIL role classifier
       ↓          ↓
  Agent        Customer
  segments     segments
       ↓          ↓
┌─────────────────────────────────┐
│  PART 3: EMOTION ANALYSIS ✅ 90%│  svar/ package: schemas, calibration,
│  Contextual Multimodal          │  turn repair, context builder, models,
│  Text + Audio + Fusion          │  acoustic baselines, trajectory, fusion
└──────────┬──────────────────────┘  Training pipeline ready, no checkpoints yet
           ↓
   TurnPrediction per segment
   (emotion, sentiment, confidence,
    evidence quality, calibration)
           ↓
┌─────────────────────────────────┐
│  PART 4: COMPLIANCE + QA  ✅    │  Gemini Flash LLM compliance,
│  + CRM                          │  30+ Hindi abuse keywords, 5-factor QA
└──────────┬──────────────────────┘  TF-IDF CRM notes
           ↓
┌─────────────────────────────────┐
│  DASHBOARD + INTEGRATION ✅ 90% │  8-tab dashboard, background pipeline,
│  Parallel pipeline + polling    │  progress tracking, parallel stages
└─────────────────────────────────┘
```

---

## Part 1 — Denoising + Enhancement Pipeline ✅ COMPLETE

All components implemented, tested, and benchmarked. Benchmarked on 3 sample calls with SNR improvements up to +9.11 dB.

### ✅ Day 1 — Audio Loader + Quality Metrics
- `denoising/audio_loader.py` — load WAV/MP3/OPUS, stereo→mono, resample to 16kHz via librosa
- `denoising/snr_calculator.py` — hybrid VAD + spectral-percentile SNR (not naive FFT)
- `denoising/clipping_detector.py` — consecutive-run clipping detection (≥3 flatline samples)
- `denoising/vad_basic.py` — RMS energy-threshold VAD with 10th-percentile noise floor

### ✅ Day 2 — Silence Ratio + Timeline Visualization
- `denoising/silence_ratio.py` — RMS-based silence ratio via VAD
- `denoising/visualize_timeline.py` — speech/silence timeline plot
- `denoising/tests/test_synthetic.py` — synthetic unit tests for all DSP components
- `denoising/tests/test.py` — batch quality assessment on real calls → `quality_report.json`

### ✅ Day 3 — Audio Enhancement Chain
- `denoising/highpass_filter.py` — 5th-order Butterworth high-pass at 80 Hz
- `denoising/notch_filter.py` — IIR notch at 50 Hz (Indian mains hum), Q=30
- `denoising/compressor.py` — RMS-based dynamic range compressor (-20 dBFS, 3:1 ratio)
- `denoising/declipper.py` — cubic spline interpolation for clipped peaks (≥0.99 threshold)
- `denoising/enhancement_pipeline.py` — chains: declip → HPF → notch → compressor
- `denoising/visualize_enhancement.py` — FFT comparison plots (0–1000 Hz focus)

### ✅ Day 4 — Spectral Wiener Denoiser
- `denoising/spectral_denoiser.py` — STFT Wiener filter (25ms Hamming, 50% overlap)
  - Noise PSD from first 0.5s silence region
  - Spectral floor (gain_floor=0.1) to prevent musical noise
  - Configurable over-subtraction factor

### ✅ Day 5 — Pipeline Integration + Benchmarks
- `denoising/pipeline.py` — `DenoiserPipeline` class
  - Returns: `snr_before_db`, `snr_after_db`, `snr_improvement_db`, `clipping_ratio`, `silence_ratio`, `hum_removed`, `compression_applied`, `audio_quality_grade`
  - Hum drop measurement via 48–52 Hz band power comparison
- `denoising/tests/benchmark.py` — batch benchmark → CSV export

**Benchmark Results:**
| File | Input SNR | Output SNR | Improvement | Grade |
|---|---|---|---|---|
| sample_audio.mp3 | 54.09 dB | 53.91 dB | -0.18 dB | PASS |
| sample_audio_2.opus | 38.02 dB | 47.13 dB | **+9.11 dB** | PASS |
| sample_audio_3.opus | 33.97 dB | 38.71 dB | **+4.73 dB** | PASS |

---

## Part 2 — Speaker Diarization + Role Inference ✅ COMPLETE

> **Architecture Change:** The original roadmap planned a custom MFCC→prosodic→formant→fingerprint pipeline. The implementation uses **pyannote/speaker-diarization-3.1** for segmentation + **SpeechBrain ECAPA-TDNN** for embeddings + **sklearn silhouette scoring** for confidence. Speaker roles are assigned via a **MuRIL-based classifier** with heuristic fallback.

### ✅ Core Diarization (Implemented via pyannote + SpeechBrain)
- `diarization/pipeline.py` — `DiarizationPipeline` class
  - Uses `pyannote/speaker-diarization-3.1` (HuggingFace, lazy-loaded)
  - Hook-based extraction of chunk embeddings + segmentation data + global centroids
  - Outputs `spk_0`/`spk_1` labels (role inference maps to agent/customer after STT)
  - Talk ratio computation (agent/customer/overlap durations)
  - Configurable `num_speakers`, `min_speakers`, `max_speakers`

- `diarization/speaker_embedder.py` — SpeechBrain ECAPA-TDNN module
  - `_embed_chunk()` / `_embed_centered()` — extract embeddings for time ranges
  - `extract_segment_embeddings()` — batch embedding extraction with min-duration padding
  - `split_merged_turns()` — detect and split merged turns where pyannote missed speaker changes
  - `reassign_speakers()` — anchor-based reclassification using SpeechBrain embeddings
  - `decode_region()` — Viterbi sequence decoder with transition priors for ambiguous regions

- `diarization/change_detector.py` — ML-based false split detection
  - `score_boundaries()` — scores each inter-segment boundary using a trained classifier
  - `merge_false_splits()` — merges adjacent same-speaker segments below confidence threshold
  - `detect_single_speaker()` — detects single-speaker audio via pyannote segmentation dominance

- `diarization/confidence.py` — embedding-based confidence scoring
  - `compute_segment_confidence()` — sklearn silhouette_samples on segment embeddings
  - `compute_rolling_separability()` — sliding-window cosine distance between speaker centroids

### ✅ Role Inference (Commits `c574531`, `6bbe8ae`)
- `svar/models/role_classifier.py` — MuRIL-based 3-class speaker role classifier:
  - Classes: `A_AGENT_B_CUSTOMER`, `A_CUSTOMER_B_AGENT`, `UNKNOWN_OR_OTHER`
  - `build_role_context()` — role-tagged transcript for classification
  - `infer_roles()` — model prediction with confidence scores
  - `apply_role_mapping()` — maps model output to segment speaker labels

- `svar/role_inference.py` — `RoleInferenceEngine` orchestrator:
  - Lazy model loading with singleton pattern
  - Classifier → heuristic → fallback chain
  - Heuristic: Hindi self-introduction pattern matching (first 12 turns only)
  - `ROLE_INFERENCE_DISABLED=1` env var to skip

### ✅ Dashboard Integration
- `diarization/dashboard_server.py` — HTTP server (port 8050) with:
  - `GET /` — serves dashboard HTML
  - `GET /api/sample_calls` — lists available audio files
  - `GET /audio/<filename>` — serves raw audio files
  - `GET /api/progress?file=X` — pipeline progress polling
  - `POST /api/analyze` — starts background pipeline
  - `POST /api/denoise`, `/api/diarize`, `/api/transcribe`, `/api/emotion`, `/api/compliance`, `/api/qa-score`, `/api/crm-note` — cached result endpoints
  - `ThreadingTCPServer` for concurrent requests

---

## Part 3 — Emotion Analysis (Contextual Multimodal) ✅ 95% COMPLETE

> **Architecture Evolution:** The original plan used DistilRoBERTa as the text backbone. The implementation uses **Contextual MuRIL** (237M params, 6-task multi-head) + **WavLM** audio encoder + **speaker-normalized acoustic baselines** + **learned gated fusion** + **temperature calibration**. The old `sentiment/` pipeline (DistilRoBERTa fallback) is retained for production use when no trained svar checkpoint exists.

### ✅ Taxonomy + Schemas (Commits `2d8bd44`, `7fe5a5f`)
- `svar/schemas.py` — canonical 16-class EmoInHindi taxonomy:
  ```
  neutral, anticipation, anger, sad, confident, fear, disgusted,
  surprised, hopeful, annoyed, compassion, joy, apprehensive,
  grateful, guilty, impressed
  ```
- `AcousticOutput` dataclass: arousal, voice_shift, escalation, confidence, source
- `SpeakerBaselineProfile` dataclass: per-speaker baseline stats (robust median + MAD)
- `EvidenceQualityGate` dataclass: text_coverage, translation_ok, audio_indeterminate, repair_used
- `IGNORE_INDEX = -100` for missing labels (interaction-state, conduct-risk not in EmoInHindi)

### ✅ Calibration + Evidence Quality (Commit `15d4e16`)
- `svar/calibration.py`:
  - Temperature scaling: single scalar `T` fitted on validation NLL
  - `EvidenceQualityGate`: deterministic quality diagnostics (separate from learned uncertainty)
  - **Abstention rule**: outputs `"uncertain"` when evidence is insufficient — NEVER outputs `"neutral"` as fallback

### ✅ Data Preparation (Commit `28517c3`)
- `svar/data/prepare_emoinhindi.py`:
  - Dialogue-level 70/15/15 split (no dialogue leakage between train/val/test)
  - Stable IDs by `_source_index` — never matched by text content
  - `repair_turns` REMOVED from training data

### ✅ Acoustic System (Commits `9256320`, `ce4ca5a`)
- `svar/acoustic/speaker_baseline.py` — `SpeakerBaselineBuilder`:
  - Early-call reference turns only (≥1.2s duration, min 3 turns, min 5s total)
  - Robust median + MAD stats, **frozen after construction**
- `svar/acoustic/baseline_features.py` — 15 low-level acoustic features
- `svar/acoustic/relative_features.py` — baseline-relative z-score vectors
- `svar/acoustic/trajectory_model.py` — `SpeakerShiftTemporalModel` (BiGRU)

### ✅ Core Models (Commit `05c97a8`)
- `svar/models/contextual_muril.py` — 237M params (195M trainable), 6-task multi-head
- `svar/models/wavlm_emotion.py` — WavLM base audio encoder with attention pooling
- `svar/models/fusion_net.py` — baseline-aware learned gated fusion
- `svar/models/temporal_decoder.py` — BiGRU across call-level turn sequence

### ✅ Pipeline + Inference (Commit `ce4ca5a`)
- `svar/pipeline.py` — `EmotionPipeline` end-to-end inference

### ✅ Turn Repair + Context Builder
- `svar/turn_repair.py` — speaker-aware merging (inference only)
- `svar/context_builder.py` — role-tagged context windows

### ✅ Training Scripts + Configs (Commit `381f505`)
- `scripts/train_muril.py`, `scripts/train_wavlm.py`, `scripts/evaluate.py`
- `configs/train_muril.yaml`, `configs/train_wavlm.yaml`, `configs/evaluate.yaml`

### ✅ Dashboard Emotion Display (Commits `d96ddb5`, `2e56379`)
- 16-class emotion chip CSS + uncertain dashed style
- `sentiment/fusion_layer.py` — neutral-override: when text=neutral but acoustic detects non-neutral, acoustic wins
- Sentiment threshold lowered from ±0.30 to ±0.10

### ✅ Old Pipeline Updated (Commit `7fe5a5f`)
- `sentiment/emotion_classifier.py` — LABEL_MAP fixed to canonical 16-class names
- `sentiment/fusion_layer.py` — EMOTION_TO_SENTIMENT updated for full 16 classes
- `sentiment/acoustic_emotion/acoustic_emotion_classifier.py` — EMOTION_PROFILES updated

### 🔴 Remaining: Training + Checkpoints
- No trained models exist yet — architecture complete but needs:
  1. Download + process EmoInHindi dataset → `python -m svar.data.prepare_emoinhindi`
  2. Train ContextualMuRIL → `python -m scripts.train_muril --config configs/train_muril.yaml`
  3. Collect real call data with human annotations for IS/CR/arousal/shift/escalation
  4. Train WavLM on EmoInHindi + fine-tune on call data
  5. Train trajectory model on annotated call data
  6. Train fusion model after text-only and audio-only validation

---

## Part 4 — Compliance + QA + CRM ✅ COMPLETE

### ✅ Compliance Engine with Gemini Flash LLM (Commits `ed5ff00`, `93fb8cc`, `bc34811`, `9fbb601`)
- `sentiment/compliance_engine.py`:
  - Two-stage architecture: keyword filter → Gemini Flash LLM verification
  - RBI/IRDAI violation regex patterns (Hindi + English)
  - 30+ Hindi abusive keywords + compound phrase detection
  - Threat/harassment pattern detection
  - Adaptive Levenshtein fuzzy match (distance 1 for words <6 chars, 2 for ≥6)
  - Hindi-aware tokenizer (`[\w\u0900-\u097F]+`)
  - WHITELIST_HINDI: 18 innocent words to prevent false positives
  - LLM overrides keyword results when contextually incorrect

- `sentiment/compliance_llm.py` — Gemini Flash compliance checker:
  - 9 API keys with round-robin rotation
  - 4-model fallback chain: `gemini-2.5-flash` → `3.6-flash` → `3.1-flash-lite` → `3.5-flash`
  - Structured prompt for Hindi/English abuse/RBI/IRDAI detection
  - `LLM_COMPLIANCE_DISABLED=1` env var to skip

### ✅ QA Scorer
- `sentiment/qa_scorer.py`:
  - 5-factor weighted scoring: customer_sentiment(0.30), compliance(0.25), agent_stability(0.20), intent_resolution(0.15), talk_ratio(0.10)
  - Grade: A(85+) B(70+) C(55+) D(<55)
  - Configurable weights in YAML

### ✅ CRM Note Generator
- `sentiment/crm_note_generator.py`:
  - TF-IDF extractive summarizer
  - Key points, compliance summary, recommended action

### ✅ Sentiment Pipeline Integration
- `sentiment/pipeline.py` — `SentimentPipeline` class
- `sentiment/stt/stt_transcriber.py` — Google Cloud STT V2 (Chirp 3), parallel chunk transcription

---

## Dashboard + Integration ✅ 90% COMPLETE

### ✅ 8-Tab Dashboard with Background Pipeline
- `dashboard/index.html` — tabs: Summary, Denoising, Diarization, Transcript, Emotion, Compliance, QA Score, CRM Note
- **Analyze button** triggers full pipeline in background
- **Progress bar** with stage-by-stage updates (polls `/api/progress`)
- Tab badges show ✓ (done) / ● (running) per stage
- Lazy result fetching after pipeline completes
- Real-time audio playhead sync across all tabs
- Emotion chips with 16-class CSS + uncertain dashed style

### ✅ Parallel Pipeline Architecture
- Background pipeline runs in daemon thread with progress tracking
- 9 stages, 2 parallel windows:
  ```
  Denoise → Diarize → [STT | Acoustic] → [Text Emotion | Compliance] → Fusion → QA → CRM
  ```
- Stage 3: STT + Acoustic run in parallel (both depend on diarize)
- Stage 4: Text Emotion + Compliance run in parallel (both depend on STT)
- Thread-safe progress updates with `_progress_lock`
- Individual endpoints remain for cached result retrieval

### ✅ Dashboard Server
- `diarization/dashboard_server.py`:
  - `POST /api/analyze` — starts background pipeline thread
  - `GET /api/progress?file=X` — returns `{status, percent, current_stage, stages, time_s}`
  - Individual POST endpoints return cached results for each tab
  - Google Cloud STT V2 (Chirp 3) with chunked 50s requests
  - Local translation model (opus-mt-hi-en) + DistilRoBERTa for Hindi emotion

---

## Integration + Polish 🔴 NOT STARTED

### 🔴 FastAPI Backend
**What needs to be built:**
- `backend/main.py` — FastAPI async endpoints
- JWT authentication middleware (HS256)
- Pydantic response models
- `/health` endpoint

### 🔴 Celery + Task Queue
**What needs to be built:**
- Celery + RabbitMQ for async pipeline execution
- Task chain: `denoise → diarize → analyze_sentiment`

### 🔴 Database + Caching
**What needs to be built:**
- MongoDB document schemas (Call, Segment, Agent)
- Redis cache for dashboard queries (TTL 5 min)

### 🔴 Docker Compose
**What needs to be built:**
- `Dockerfile` for API + worker
- `docker-compose.yml`: api, worker, rabbitmq, mongodb, redis

### 🔴 README + Documentation
**What needs to be built:**
- Professional README with architecture diagram, benchmark tables
- Setup instructions: `git clone → docker-compose up`

---

## Summary

| Part | Status | Key Components |
|---|---|---|
| Denoising + Enhancement | ✅ Complete | Butterworth HPF, IIR notch, compressor, declipper, Wiener denoiser |
| Diarization + Role Inference | ✅ Complete | pyannote 3.1, SpeechBrain ECAPA-TDNN, MuRIL role classifier, heuristic fallback |
| Emotion Analysis | ✅ 95% | svar/ package: 16-class taxonomy, contextual MuRIL, WavLM, acoustic baselines, fusion, calibration |
| Compliance + QA + CRM | ✅ Complete | Gemini Flash LLM, Hindi abuse lexicon, 5-factor QA, TF-IDF CRM notes |
| Dashboard + Integration | ✅ 90% | 8-tab UI, background pipeline, progress tracking, parallel stages |
| Integration + Polish | 🔴 0% | FastAPI, Celery, MongoDB, Docker Compose |

### Completed Components (40+ files)
```
svar/schemas.py                    svar/pipeline.py
svar/calibration.py                svar/turn_repair.py
svar/context_builder.py            svar/data/prepare_emoinhindi.py
svar/models/contextual_muril.py    svar/models/wavlm_emotion.py
svar/models/fusion_net.py          svar/models/temporal_decoder.py
svar/models/role_classifier.py     svar/role_inference.py
svar/acoustic/speaker_baseline.py  svar/acoustic/baseline_features.py
svar/acoustic/relative_features.py svar/acoustic/trajectory_model.py

denoising/audio_loader.py          denoising/pipeline.py
denoising/snr_calculator.py        denoising/enhancement_pipeline.py
denoising/clipping_detector.py     denoising/highpass_filter.py
denoising/silence_ratio.py         denoising/notch_filter.py
denoising/vad_basic.py             denoising/compressor.py
denoising/declipper.py             denoising/spectral_denoiser.py

diarization/pipeline.py            diarization/speaker_embedder.py
diarization/change_detector.py     diarization/confidence.py
diarization/dashboard_server.py    dashboard/index.html

sentiment/emotion_classifier.py    sentiment/fusion_layer.py
sentiment/compliance_engine.py     sentiment/compliance_llm.py
sentiment/acoustic_emotion/acoustic_pipeline.py
sentiment/stt/stt_transcriber.py
sentiment/qa_scorer.py             sentiment/crm_note_generator.py
sentiment/pipeline.py

scripts/train_muril.py             scripts/train_wavlm.py
scripts/evaluate.py                docs/target_domain_annotation.md
configs/train_muril.yaml           configs/train_wavlm.yaml
configs/evaluate.yaml
```

### Remaining Work (priority order)
1. **EmoInHindi Data** — download + process → `python -m svar.data.prepare_emoinhindi`
2. **Train ContextualMuRIL** — text model → `python -m scripts.train_muril`
3. **Collect Annotated Call Data** — IS/CR/arousal/shift/escalation labels
4. **Train WavLM** — audio model on EmoInHindi + fine-tune on call data
5. **Train Trajectory Model** — on annotated call data
6. **Train Fusion Model** — after text-only and audio-only show useful signal
7. **FastAPI Backend** — replace http.server with proper async API
8. **Docker Compose** — containerized deployment

---

## Dataset Citations

```
EmoInHindi (LREC 2022) — IIT Patna
Singh et al., "EmoInHindi: A Multi-label Emotion and Intensity Annotated
Dataset in Hindi for Emotion Recognition in Dialogues"
44,247 utterances, 16 emotions, Fleiss κ=0.84

Kaggle Hindi Call Center Audio — @infobayai
760+ hours of dual-channel Hindi telephony recordings

Vaani Corpus — IISc Bangalore (Gnani.ai's academic partner)
150,000+ hours, 773 districts, 22+ Indian languages
```

---

## Architecture Notes

### Why Contextual MuRIL instead of DistilRoBERTa?

The implementation uses **Contextual MuRIL** because:
1. **Native Hindi understanding** — pre-trained on 11 Indian languages including Hindi
2. **Context windows** — role-tagged context captures conversational dynamics
3. **Multi-task learning** — 6 heads share representations
4. **237M params** (195M trainable) — significantly larger capacity than DistilRoBERTa (82M)

### Why Speaker-Normalized Acoustic Features?

1. **Speaker variability** — pitch, energy, speaking rate vary dramatically between speakers
2. **Baseline-relative features** — z-score deviation captures CHANGE, not absolute state
3. **Frozen baselines** — early-call only, never updated with later turns
4. **Robust statistics** — median + MAD to handle outliers

### Why Learned Fusion instead of Confidence-Gated?

1. **Text-dominates-neutral override** — acoustic wins when text=neutral but acoustic detects non-neutral
2. **Audio provides CHANGE signal** — arousal/shift/escalation modulate text emotion
3. **Confidence modulator** — acoustic shift boosts confidence when text is uncertain
4. **Escalation detector** — sustained acoustic change overrides low-confidence text

### Why Gemini Flash for Compliance?

1. **Context understanding** — LLM distinguishes "कमीज" (shirt) from abusive Hindi words
2. **False positive suppression** — LLM overrides keyword matches when contextually wrong
3. **Multi-language** — handles Hindi+English code-switching in abuse detection
4. **Speed** — Gemini Flash responds in <1s for compliance checks

### Why Parallel Pipeline?

The pipeline has natural parallelism:
- **STT + Acoustic** are independent after diarization (different data dependencies)
- **Text Emotion + Compliance** are independent after STT (both only need text)
- Parallel execution reduces total pipeline time by ~30% on multi-core systems

---

## Commits (20 total)
```
6bbe8ae fix: role resolution false positive + diarization tab crash
c574531 feat: transcript-based role classifier replaces static first-speaker=agent
9fbb601 feat: replace regex compliance with Gemini Flash LLM
bc34811 fix: compliance engine false positives from fuzzy matching
cbfd82a perf: parallel STT chunk transcription (5x) + concurrent acoustic/text emotion analysis
a68cbb3 fix: compliance/qa/crm API endpoints return wrapped results matching frontend expectations
fc0f685 docs: update roadmap with compliance/QA/CRM completion
93fb8cc compliance: remove hell/damn/crap from English abuse list
ed5ff00 compliance: expand Hindi abuse lexicon, compound patterns, threat detection
97e566e docs: update roadmap to reflect svar package, acoustic system, fusion improvements
2e56379 fusion: neutral-override when text=neutral but acoustic disagrees
f06bbee data: add init
d96ddb5 dashboard: EmoInHindi 16-class emotion chip CSS + uncertain style
05c97a8 core: schemas, models, turn repair, context builder, annotation docs
381f505 training: scripts + YAML configs for muril, wavlm, evaluation
ce4ca5a fusion: baseline-aware audio change signal, pipeline: real audio inference
9256320 acoustic: speaker baseline, features, relative vectors, trajectory model
28517c3 data: stable IDs, no repair_turns, IGNORE_INDEX for missing labels
15d4e16 calibration: uncertain not neutral, separate evidence quality gate
2d8bd44 schemas: canonical taxonomy, AcousticOutput, IGNORE_INDEX, evidence quality
7fe5a5f fix: canonical EmoInHindi 16-class taxonomy across all classifiers
```

---

## Minimum Viable Checkpoint — REACHED ✅

Parts 1 + 2 are fully functional: raw audio → denoised audio → diarized segments with confidence scores + talk ratios + web dashboard. This is a deployable proof-of-concept.

## Emotion Analysis Checkpoint — REACHED ✅

Part 3 architecture is complete: schemas, calibration, turn repair, context builder, models (contextual MuRIL, WavLM, fusion, temporal), acoustic baselines + trajectory, training pipeline, evaluation. Ready for training once EmoInHindi data is downloaded and processed.

## Compliance + QA + CRM Checkpoint — REACHED ✅

Full compliance engine with Gemini Flash LLM + keyword fallback, Hindi abuse lexicon (30+ words), compound phrase detection, threat/harassment patterns, RBI/IRDAI regex, Levenshtein fuzzy matching. QA scorer with 5-factor weighted scoring. CRM note generator with TF-IDF extractive summarization.

## Parallel Pipeline Checkpoint — REACHED ✅

Background pipeline with progress tracking, 9 stages with 2 parallel windows, Analyze button with progress bar, 8-tab dashboard with stage badges. All analysis runs server-side in background thread; tabs display cached results.
