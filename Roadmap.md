# SVAR — Call Analytics Pipeline Roadmap
*Updated to reflect current codebase state — 24 Jul 2026*

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
│  PART 3: SENTIMENT     🔶 60%   │  Preprocessing, acoustic classifier,
│  ANALYSIS (Voice+Text)          │  STT, model architecture done
└──────────┬──────────────────────┘  Training, fusion, QA remaining
           ↓
   Per-speaker emotion
   timeline + QA scores
           ↓
┌─────────────────────────────────┐
│  INTEGRATION + POLISH   🔴 TODO │  FastAPI, dashboard, Docker
│  API + Dashboard + Deploy       │
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
- `diarization/dashboard/index.html` — web UI for diarization visualization

**Pipeline output structure:**
```json
{
  "talk_ratio": {
    "agent_duration_s": 12.5,
    "customer_duration_s": 8.3,
    "overlap_duration_s": 1.2,
    "agent_ratio": 0.6019,
    "customer_ratio": 0.3981
  },
  "segments": [
    {
      "start_time_s": 0.0, "end_time_s": 2.5,
      "speaker": "agent",
      "confidence": 0.87,
      "uncertain": false,
      "sb_d_agent": 0.12,
      "sb_d_customer": 0.45,
      "sb_margin": 0.33
    }
  ],
  "separability": [...],
  "low_separability_regions": [...],
  "confidence_method": "silhouette",
  "denoise_metrics": {...}
}
```

---

## Part 3 — Sentiment Analysis 🔶 60% COMPLETE

### ✅ Day 12 — EmoInHindi Dataset Preprocessing
- `sentiment/preprocessing/prepare_emoin_hindi.py`
  - 16→6 emotion taxonomy mapping (anger, sadness, fear, happiness, disgust, neutral)
  - Intensity-based primary label selection from multi-label annotations
  - 2-utterance context window with `[SEP]` tokens
  - Dialogue-level 80/10/10 split (no dialogue leakage)
  - Neutral cap at 10,000 samples

### ✅ Day 13 — Acoustic Emotion Classifier
- `sentiment/acoustic_emotion/delta_normalizer.py`
  - Z-score delta computation: `delta = (current - baseline) / (baseline_std + ε)`
  - Maps 5 prosodic features → standardized deltas (pitch_d, energy_d, rate_d, jitter_d, pause_r)
- `sentiment/acoustic_emotion/acoustic_emotion_classifier.py`
  - `EMOTION_PROFILES` dict with 7 emotion profiles from Yildirim 2004, Cao 2014, PLOS ONE 2025
  - L2-distance similarity scoring + HiF0 position bonus (+0.30)
  - Softmax probability normalization, indeterminate flag at <0.30 confidence

### ✅ Day 14 — Whisper STT Integration
- `sentiment/stt/stt_transcriber.py`
  - `SpeechToTextTranscriber` class with lazy HuggingFace pipeline init
  - Supports local Vaani Hindi model (`whisper-hindi/`) or online `ARTPARK-IISc/whisper-large-v3-vaani-hindi`
  - Automatic resampling to 16kHz, forced Hindi decoder
  - Batch `transcribe_segments()` for segment lists
  - Fallback mode for offline testing

### ✅ Day 15 — MuRIL Dataset + Model Architecture
- `sentiment/models/dataset.py`
  - `EmotionDataset` — PyTorch Dataset with tokenizer integration, emotion/sentiment/intensity labels
  - `EMOTION_LABEL2ID` (6 classes), `SENTIMENT_LABEL2ID` (3 classes: pos/neu/neg)
  - `compute_class_weights()` — inverse-frequency weight tensor
- `sentiment/models/muril_model.py`
  - `MultiTaskMuRIL` — multi-head architecture on `google/muril-base-cased`
  - 3 heads: emotion (6), sentiment (3), intensity (1)
  - Supports local model dir (`muril-base/`) or HuggingFace
  - Fallback zero-projection when encoder unavailable

### 🔴 Day 16 — MuRIL Training Loop (NOT DONE)
**What needs to be built:**
- `sentiment/models/trainer.py` — full training loop:
  - AdamW optimizer (lr=2e-5, weight_decay=0.01)
  - Linear warmup scheduler (10% warmup steps)
  - Weighted CrossEntropyLoss for emotion head
  - MSE loss for intensity head
  - Mixed-precision (AMP) if GPU available
  - Gradient checkpointing for RTX 2050 VRAM constraints
  - Batch size 16, 5 epochs
  - Early stopping on validation macro-F1
  - Checkpoint saving (best by val macro-F1)
- Training metrics logging → CSV
- Expected: val_macro_f1 ≈ 0.72–0.74 by epoch 5

### 🔴 Day 17 — Fusion Layer + Compliance Engine (NOT DONE)
**What needs to be built:**
- `sentiment/fusion_layer.py`:
  - Confidence-gated fusion: text_conf > 0.70 → trust text, acoustic_conf > 0.65 → trust acoustic, else weighted blend (0.55 text + 0.45 acoustic)
- `sentiment/compliance_engine.py`:
  - RBI/IRDAI violation regex patterns (Hindi + English)
  - Abusive language keyword list
  - Levenshtein distance (DP from scratch)
  - Fuzzy match: flag if Levenshtein ≤ 2 against violation keywords

### 🔴 Day 18 — QA Scoring + CRM Note Generator (NOT DONE)
**What needs to be built:**
- `sentiment/qa_scorer.py`:
  ```
  score = 0.30 * customer_sentiment + 0.25 * (1 - compliance_ratio)
        + 0.20 * agent_stability + 0.15 * intent_resolution + 0.10 * talk_ratio
  ```
  - Grade: A(85+) B(70+) C(55+) D(<55)
  - Configurable weights in YAML
- `sentiment/crm_note_generator.py`:
  - Gemini 1.5 Flash API (primary) or TF-IDF extractive summarizer (fallback)
  - Input: transcript + emotion timeline + intent + compliance flags

### 🔴 Day 19 — SentimentPipeline Integration (NOT DONE)
**What needs to be built:**
- `sentiment/pipeline.py` — `SentimentPipeline` class:
  - Consumes Part 2 output (segments with speaker labels)
  - Runs: acoustic emotion classification → STT → MuRIL text classification → fusion → compliance → QA scoring
  - Returns per-segment emotion timeline + per-call QA summary
- End-to-end test on all 10 sample calls
- Write `README_part3.md` with paper citations

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
| Sentiment Analysis | 🔶 60% | 12–19 | EmoInHindi preprocessing, acoustic classifier, Whisper STT, MuRIL architecture (training TBD) |
| Integration + Polish | 🔴 0% | 20–25 | FastAPI, Celery, MongoDB, Docker Compose |

### Completed Components (20 files)
```
denoising/audio_loader.py          denoising/pipeline.py
denoising/snr_calculator.py        denoising/enhancement_pipeline.py
denoising/clipping_detector.py     denoising/highpass_filter.py
denoising/silence_ratio.py         denoising/notch_filter.py
denoising/vad_basic.py             denoising/compressor.py
denoising/declipper.py             denoising/spectral_denoiser.py
diarization/pipeline.py            diarization/speaker_embedder.py
diarization/change_detector.py     diarization/confidence.py
diarization/dashboard_server.py    sentiment/preprocessing/prepare_emoin_hindi.py
sentiment/acoustic_emotion/delta_normalizer.py
sentiment/acoustic_emotion/acoustic_emotion_classifier.py
sentiment/stt/stt_transcriber.py
sentiment/models/dataset.py        sentiment/models/muril_model.py
```

### Remaining Work (priority order)
1. **SentimentPipeline** — wire acoustic + text branches into end-to-end sentiment analysis
2. **MuRIL Training** — training loop, checkpointing, evaluation metrics
3. **Fusion Layer** — confidence-gated acoustic+text fusion
4. **Compliance Engine** — RBI/IRDAI regex, abusive language detection, Levenshtein fuzzy match
5. **QA Scoring** — weighted formula producing per-call grades
6. **CRM Note Generator** — Gemini API or TF-IDF fallback
7. **FastAPI Backend** — replace http.server with proper async API
8. **Dashboard UI** — enhance existing HTML dashboard with emotion timelines + QA scores
9. **Docker Compose** — containerized deployment
10. **Documentation** — comprehensive README with benchmarks

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

### Why pyannote + SpeechBrain instead of custom DSP diarization?

The original roadmap planned a handcrafted pipeline: MFCC extraction → LPC formant estimation → cosine fingerprinting → rolling baseline updates. During implementation, this was replaced with:

1. **pyannote/speaker-diarization-3.1** — state-of-the-art neural diarization model, handles overlapping speech, varying recording conditions, and accent diversity far better than cosine similarity on handcrafted features
2. **SpeechBrain ECAPA-TDNN** — neural speaker embeddings that outperform MFCC+formant fingerprints by a large margin on speaker verification benchmarks
3. **Sklearn silhouette scoring** — principled embedding-based confidence instead of ad-hoc "similarity difference < 0.05" thresholds

The custom components (MFCC extractor, prosodic extractor, LPC formant estimator, pause segmenter, speaker fingerprinter, speaker assigner, baseline builder) were committed earlier in the git history but are no longer imported by the active pipeline. They remain in the codebase as reference implementations.

The key additions over pyannote's default output are:
- **Merged turn splitting** — detects when pyannote assigns both speakers to one long turn
- **Single-speaker detection** — handles mono-speaker calls without false splits
- **Viterbi sequence decoding** — smooths ambiguous regions using transition priors
- **Rolling separability curve** — identifies time regions where speaker separation is unreliable
- **False split merging** — ML classifier trained to distinguish real speaker changes from over-segmentation

---

## Minimum Viable Checkpoint — REACHED ✅

Parts 1 + 2 are fully functional: raw audio → denoised audio → diarized segments with confidence scores + talk ratios + web dashboard. This is a deployable proof-of-concept.
