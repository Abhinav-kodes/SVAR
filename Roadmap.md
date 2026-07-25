# SVAR — Call Analytics Pipeline Roadmap
*Updated to reflect current codebase state — 25 Jul 2026*

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
│  + SPEAKER PROFILING            │  + SpeechBrain ECAPA-TDNN embeddings
└──────┬──────────┬───────────────┘  + silhouette confidence scoring
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
│  DASHBOARD + INTEGRATION 🔶 75% │  10-tab dashboard, lazy per-tab APIs,
│  Real-time visualization        │  emotion chips, acoustic panels
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

## Part 2 — Speaker Diarization ✅ COMPLETE

> **Architecture Change:** The original roadmap planned a custom MFCC→prosodic→formant→fingerprint pipeline. The implementation uses **pyannote/speaker-diarization-3.1** for segmentation + **SpeechBrain ECAPA-TDNN** for embeddings + **sklearn silhouette scoring** for confidence. This is significantly more robust and production-ready.

### ✅ Day 6–9 — Core Diarization (Implemented via pyannote + SpeechBrain)

**Original plan:** Custom MFCC extractor, prosodic extractor, LPC formant estimator, pause segmenter, speaker fingerprinter, speaker assigner, baseline builder.

**Actual implementation:**
- `diarization/pipeline.py` — `DiarizationPipeline` class
  - Uses `pyannote/speaker-diarization-3.1` (HuggingFace, lazy-loaded)
  - Hook-based extraction of chunk embeddings + segmentation data + global centroids
  - Automatic agent/customer assignment (first pyannote speaker = agent)
  - Talk ratio computation (agent/customer/overlap durations)
  - Configurable `num_speakers`, `min_speakers`, `max_speakers`

- `diarization/speaker_embedder.py` — SpeechBrain ECAPA-TDNN module
  - `_embed_chunk()` / `_embed_centered()` — extract embeddings for time ranges
  - `extract_segment_embeddings()` — batch embedding extraction with min-duration padding
  - `split_merged_turns()` — detect and split merged turns where pyannote missed speaker changes (drift threshold 0.30)
  - `reassign_speakers()` — anchor-based reclassification using SpeechBrain embeddings
  - `decode_region()` — Viterbi sequence decoder with transition priors for ambiguous regions

- `diarization/change_detector.py` — ML-based false split detection
  - `score_boundaries()` — scores each inter-segment boundary using a trained classifier (joblib model)
  - `merge_false_splits()` — merges adjacent same-speaker segments below confidence threshold
  - `detect_single_speaker()` — detects single-speaker audio via pyannote segmentation dominance

- `diarization/confidence.py` — embedding-based confidence scoring
  - `compute_segment_confidence()` — sklearn silhouette_samples on segment embeddings
  - `compute_rolling_separability()` — sliding-window cosine distance between speaker centroids
  - `detect_low_separability_regions()` — identifies ambiguous time regions

### ✅ Day 10–11 — Integration + Dashboard

- `diarization/dashboard_server.py` — HTTP server (port 8050) with:
  - `GET /` — serves dashboard HTML
  - `GET /api/sample_calls` — lists available audio files
  - `GET /audio/<filename>` — serves raw audio files
  - `POST /api/diarize` — full pipeline: load → denoise → diarize → return JSON
  - `POST /api/text_emotion` — text emotion via translate + DistilRoBERTa (production fallback)
  - `POST /api/acoustic_emotion` — rule-based prosodic classifier
  - `POST /api/fused_emotion` — confidence-gated text+acoustic fusion
  - All API endpoints use `ThreadingTCPServer` for concurrent requests
- `dashboard/index.html` — 10-tab web UI with lazy per-tab APIs

---

## Part 3 — Emotion Analysis (Contextual Multimodal) ✅ 90% COMPLETE

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
  - Quality gate checks: text coverage, translation quality, audio indeterminacy, repair used

### ✅ Data Preparation (Commit `28517c3`)
- `svar/data/prepare_emoinhindi.py`:
  - Dialogue-level 70/15/15 split (no dialogue leakage between train/val/test)
  - Stable IDs by `_source_index` — never matched by text content
  - `repair_turns` REMOVED from training data — each original utterance = one supervised target
  - `IGNORE_INDEX=-100` for interaction-state and conduct-risk (not in EmoInHindi)
  - Labels carried by stable `_source_index`, never matched by text

### ✅ Acoustic System (Commits `9256320`, `ce4ca5a`)
- `svar/acoustic/speaker_baseline.py` — `SpeakerBaselineBuilder`:
  - Early-call reference turns only (≥1.2s duration, min 3 turns, min 5s total)
  - Robust median + MAD stats for all acoustic features
  - **Frozen after construction** — never updated with later (potentially angry) turns
  - Prevents emotional drift in baseline estimates

- `svar/acoustic/baseline_features.py` — 15 low-level acoustic features:
  - log-energy, F0 median, F0 IQR, F0 slope, voiced ratio
  - speech rate, pause ratio, spectral centroid, spectral slope
  - HNR (harmonics-to-noise ratio), jitter, shimmer

- `svar/acoustic/relative_features.py` — baseline-relative vectors:
  - Z-score deviation from baseline per feature
  - Delta between consecutive turns
  - Scalar shift scores for arousal, voice_shift, escalation

- `svar/acoustic/trajectory_model.py` — `SpeakerShiftTemporalModel`:
  - BiGRU across call-level turn sequence
  - Outputs: arousal, voice_shift, escalation
  - Trained only on real call labels (not EmoInHindi — which lacks these)
  - Provides CHANGE SIGNAL, not competing emotion labels

### ✅ Core Models (Commit `05c97a8`)
- `svar/models/contextual_muril.py` — 237M params (195M trainable):
  - Multi-task heads: emotion BCE + intensity SmoothL1 + sentiment CE + interaction-state CE + conduct-risk BCE + uncertainty
  - Role-tagged context windows: `[AGENT] prev [CUSTOMER] prev [TARGET_AGENT] target ...`
  - Dropout regularization per task head
  - `IGNORE_INDEX` mask support for missing labels

- `svar/models/wavlm_emotion.py` — WavLM base audio encoder:
  - Gradient checkpointing for RTX 2050 VRAM constraints
  - Attention pooling over frame-level features
  - Speaker-normalized input (relative to baseline)

- `svar/models/fusion_net.py` — baseline-aware learned fusion:
  - Audio provides CHANGE SIGNAL (arousal/shift/escalation), not competing emotion
  - Gated fusion: text determines WHAT emotion, audio determines HOW INTENSE
  - Confidence modulator: acoustic shift boosts confidence when text is uncertain
  - Escalation detector: sustained acoustic change overrides low-confidence text

- `svar/models/temporal_decoder.py` — BiGRU/Transformer across call-level turn sequence
  - Captures escalation patterns across the conversation

### ✅ Pipeline + Inference (Commit `ce4ca5a`)
- `svar/pipeline.py` — `EmotionPipeline` end-to-end:
  ```
  segments → turns → baselines → relative features → trajectory model →
  text model → fusion → calibration → TurnPrediction
  ```
  - `TurnPrediction` dataclass: emotion, sentiment, intensity, confidence, calibration_temp, evidence_quality, acoustic
  - Full inference verified working on real audio (71 segments, ~195s)

### ✅ Turn Repair + Context Builder
- `svar/turn_repair.py` — speaker-aware merging:
  - Gap threshold: 0.7s, short-turn threshold: 0.8s
  - **INFERENCE ONLY** — never run on training data
  - Each original EmoInHindi utterance = one supervised target

- `svar/context_builder.py` — role-tagged context:
  - `[AGENT] prev [CUSTOMER] prev [TARGET_AGENT] target ...`
  - Configurable context window size

### ✅ Training Scripts + Configs (Commit `381f505`)
- `scripts/train_muril.py` — multi-task training:
  - AdamW optimizer (lr=2e-5, weight_decay=0.01)
  - Linear warmup scheduler (10% warmup steps)
  - Weighted BCE for emotion, SmoothL1 for intensity, CE for sentiment
  - Mixed-precision (AMP) + gradient checkpointing
  - Early stopping on validation macro-F1
  - Checkpoint saving (best by val macro-F1)

- `scripts/train_wavlm.py` — WavLM audio model training
- `scripts/evaluate.py` — call-disjoint evaluation metrics
- `configs/train_muril.yaml`, `configs/train_wavlm.yaml`, `configs/evaluate.yaml`

### ✅ Annotation Schema
- `docs/target_domain_annotation.md` — human annotation schema for:
  - Interaction-state: professional, escalatory, defensive, cooperative
  - Conduct-risk: clean, warning, violation, critical
  - Acoustic labels: arousal, shift, escalation (continuous 0-1)

### ✅ Dashboard Emotion Display (Commits `d96ddb5`, `2e56379`)
- 16-class emotion chip CSS in `dashboard/index.html`
- Uncertain style: dashed border for uncertain predictions
- `sentiment/fusion_layer.py` — neutral-override: when text=neutral but acoustic detects non-neutral, acoustic wins
- Sentiment threshold lowered from ±0.30 to ±0.10 for better sensitivity
- All existing classifiers updated to canonical 16-class taxonomy

### ✅ Old Pipeline Updated (Commit `7fe5a5f`)
- `sentiment/emotion_classifier.py` — LABEL_MAP fixed to canonical 16-class names
- `sentiment/fusion_layer.py` — EMOTION_TO_SENTIMENT updated for full 16 classes
- `sentiment/acoustic_emotion/acoustic_emotion_classifier.py` — EMOTION_PROFILES updated: removed "stress", renamed to canonical names

### 🔴 Remaining: Training + Checkpoints
- No trained models exist yet — architecture complete but needs:
  1. Download + process EmoInHindi dataset → `python -m svar.data.prepare_emoinhindi`
  2. Train ContextualMuRIL → `python -m scripts.train_muril --config configs/train_muril.yaml`
  3. Collect real call data with human annotations for IS/CR/arousal/shift/escalation
  4. Train WavLM on EmoInHindi + fine-tune on call data
  5. Train trajectory model on annotated call data
  6. Train fusion model after text-only and audio-only validation

---

## Dashboard + Integration 🔶 75% COMPLETE

### ✅ 10-Tab Dashboard
- `dashboard/index.html` — tabs: Transcript, Diarization, Emotion Timeline, Acoustic, QA Score, Compliance, CRM Notes, Agent Profile, Raw JSON, Settings
- Lazy per-tab API loading (only fetches when tab opened)
- Real-time transcript highlighting with speaker colors
- Emotion chips with 16-class CSS + uncertain dashed style

### ✅ Dashboard Server
- `diarization/dashboard_server.py`:
  - `ThreadingTCPServer` for concurrent requests (replaces `HTTPServer`)
  - All endpoints: `/api/diarize`, `/api/text_emotion`, `/api/acoustic_emotion`, `/api/fused_emotion`, `/api/qa_score`, `/api/compliance`
  - Google Cloud STT V2 (Chirp 3) with chunked 50s requests
  - Local translation model (opus-mt-hi-en) + DistilRoBERTa for Hindi emotion

### 🔴 Remaining: Integration
- Wire `api_text_emotion` to `svar.pipeline.EmotionPipeline` (with fallback to existing classifier when no checkpoint)
- Wire acoustic panel to `svar/acoustic/` system (requires trained WavLM + trajectory model)

---

## Integration + Polish 🔴 NOT STARTED

### 🔴 Day 20 — FastAPI Backend
**What needs to be built:**
- `backend/main.py` — FastAPI async endpoints:
  ```
  POST /calls/upload           → accept audio file, return job_id
  GET  /calls/{job_id}/status  → poll for completion
  GET  /calls/{job_id}/results → full analysis JSON
  GET  /dashboard/agent/{id}   → agent QA summary
  GET  /dashboard/overview     → all agents ranked by score
  ```
- JWT authentication middleware (HS256)
- Pydantic response models
- `/health` endpoint
- Replace current `http.server` dashboard with FastAPI

### 🔴 Day 21 — Celery + Task Queue
**What needs to be built:**
- Celery + RabbitMQ for async pipeline execution
- Task chain: `denoise → diarize → analyze_sentiment`
- Background job processing with status polling

### 🔴 Day 22 — Database + Caching
**What needs to be built:**
- MongoDB document schemas (Call, Segment, Agent)
- Full analysis persistence per call
- Redis cache for dashboard queries (TTL 5 min)
- Rolling 30-call window for agent QA tracking

### 🔴 Day 23 — Docker Compose
**What needs to be built:**
- `Dockerfile` for API + worker
- `docker-compose.yml`: api, worker, rabbitmq, mongodb, redis
- `.env` for API keys and connection strings
- Health checks, horizontal scaling

### 🔴 Day 24 — README + Documentation
**What needs to be built:**
- Professional README with architecture diagram, benchmark tables, sample output JSON
- Methodology section citing all papers
- Setup instructions: `git clone → docker-compose up`

### 🔴 Day 25 — Final Testing + Demo
**What needs to be built:**
- 20-call zero-crash validation
- 3-minute demo video
- GitHub push with clean history

---

## Summary

| Part | Status | Days | Key Components |
|---|---|---|---|
| Denoising + Enhancement | ✅ Complete | 1–5 | Butterworth HPF, IIR notch, compressor, declipper, Wiener denoiser, SNR calculator |
| Diarization + Profiling | ✅ Complete | 6–11 | pyannote 3.1, SpeechBrain ECAPA-TDNN, silhouette confidence, Viterbi decoder, change detector |
| Emotion Analysis | ✅ 90% | 12–19 | svar/ package: 16-class taxonomy, contextual MuRIL, WavLM, acoustic baselines, trajectory model, learned fusion, calibration |
| Dashboard + Integration | 🔶 75% | — | 10-tab dashboard, lazy APIs, emotion chips, Google Cloud STT |
| Integration + Polish | 🔴 0% | 20–25 | FastAPI, Celery, MongoDB, Docker Compose |

### Completed Components (35+ files)
```
svar/schemas.py                    svar/pipeline.py
svar/calibration.py                svar/turn_repair.py
svar/context_builder.py            svar/data/prepare_emoinhindi.py
svar/models/contextual_muril.py    svar/models/wavlm_emotion.py
svar/models/fusion_net.py          svar/models/temporal_decoder.py
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
sentiment/acoustic_emotion/acoustic_emotion_classifier.py
sentiment/preprocessing/prepare_emoin_hindi.py
sentiment/stt/stt_transcriber.py
sentiment/models/dataset.py        sentiment/models/muril_model.py
sentiment/compliance_engine.py     sentiment/qa_scorer.py
sentiment/crm_note_generator.py    sentiment/pipeline.py

scripts/train_muril.py             scripts/train_wavlm.py
scripts/evaluate.py                docs/target_domain_annotation.md
configs/train_muril.yaml           configs/train_wavlm.yaml
configs/evaluate.yaml
```

### Remaining Work (priority order)
1. **EmoInHindi Data** — download + process → `python -m svar.data.prepare_emoinhindi`
2. **Train ContextualMuRIL** — text model → `python -m scripts.train_muril`
3. **Wire Dashboard to svar** — `api_text_emotion` → `svar.pipeline.EmotionPipeline` (fallback when no checkpoint)
4. **Collect Annotated Call Data** — IS/CR/arousal/shift/escalation labels (needed for trajectory + fusion training)
5. **Train WavLM** — audio model on EmoInHindi + fine-tune on call data
6. **Train Trajectory Model** — on annotated call data
7. **Train Fusion Model** — after text-only and audio-only show useful signal
8. **FastAPI Backend** — replace http.server with proper async API
9. **Celery + Task Queue** — async pipeline execution
10. **Docker Compose** — containerized deployment

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

The original plan used DistilRoBERTa as the text backbone (via HuggingFace pipeline). The implementation uses **Contextual MuRIL** (Multilingual Representations for Indian Languages) because:

1. **Native Hindi understanding** — MuRIL is pre-trained on 11 Indian languages including Hindi, handling code-switching (Hindi+English) that DistilRoBERTa misses
2. **Context windows** — role-tagged context (`[AGENT] prev [CUSTOMER] prev`) captures conversational dynamics
3. **Multi-task learning** — 6 heads (emotion, intensity, sentiment, interaction-state, conduct-risk, uncertainty) share representations
4. **237M params** (195M trainable) — significantly larger capacity than DistilRoBERTa (82M)

### Why Speaker-Normalized Acoustic Features?

The acoustic system uses **speaker baselines** (early-call reference turns) instead of absolute features because:

1. **Speaker variability** — pitch, energy, speaking rate vary dramatically between speakers
2. **Baseline-relative features** — z-score deviation from each speaker's own baseline captures CHANGE, not absolute state
3. **Frozen baselines** — early-call only (≥1.2s, min 3 turns, min 5s), never updated with later (potentially angry) turns
4. **Robust statistics** — median + MAD (not mean + std) to handle outliers

### Why Learned Fusion instead of Confidence-Gated?

The original plan used simple confidence-threshold gating. The implementation uses **learned gated fusion** because:

1. **Text-dominates-neutral override** — when text=neutral but acoustic detects non-neutral, acoustic wins (fixes Hindi profanity mistranslation)
2. **Audio provides CHANGE signal** — arousal/shift/escalation modulate text emotion, not compete with it
3. **Confidence modulator** — acoustic shift boosts confidence when text is uncertain
4. **Escalation detector** — sustained acoustic change overrides low-confidence text

### Why pyannote + SpeechBrain instead of custom DSP diarization?

The original roadmap planned a handcrafted pipeline: MFCC extraction → LPC formant estimation → cosine fingerprinting → rolling baseline updates. During implementation, this was replaced with:

1. **pyannote/speaker-diarization-3.1** — state-of-the-art neural diarization model, handles overlapping speech, varying recording conditions, and accent diversity far better than cosine similarity on handcrafted features
2. **SpeechBrain ECAPA-TDNN** — neural speaker embeddings that outperform MFCC+formant fingerprints by a large margin on speaker verification benchmarks
3. **Sklearn silhouette scoring** — principled embedding-based confidence instead of ad-hoc "similarity difference < 0.05" thresholds

The key additions over pyannote's default output are:
- **Merged turn splitting** — detects when pyannote assigns both speakers to one long turn
- **Single-speaker detection** — handles mono-speaker calls without false splits
- **Viterbi sequence decoding** — smooths ambiguous regions using transition priors
- **Rolling separability curve** — identifies time regions where speaker separation is unreliable
- **False split merging** — ML classifier trained to distinguish real speaker changes from over-segmentation

---

## Commits (11 total)
```
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
