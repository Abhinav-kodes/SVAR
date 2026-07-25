# SVAR

**Sentiment and Voice Analytics for Real-time QA** — an end-to-end Hindi call analytics pipeline that denoises recordings, separates speakers, transcribes speech, infers roles, detects emotions, enforces compliance, scores agent quality, and generates CRM notes.

---

## Architecture

```
Raw Audio
  │
  ▼
┌──────────────┐
│  Denoising   │  SNR improvement, declipping, spectral denoising
└──────┬───────┘
       ▼
┌──────────────┐
│ Diarization  │  pyannote speaker segmentation + SpeechBrain embeddings
└──────┬───────┘
       ▼
┌──────────────────────────────────────────┐
│  Parallel Stage                           │
│  ┌────────────┐  ┌───────────────────┐    │
│  │  STT (Chirp│  │  Acoustic Baseline│    │
│  │  3 V2)     │  │  (speaker-normalized) │
│  └─────┬──────┘  └────────┬──────────┘    │
└────────┼──────────────────┼───────────────┘
         ▼                  ▼
┌──────────────────────────────────────────┐
│  Parallel Stage                           │
│  ┌────────────┐  ┌───────────────────┐    │
│  │  Text      │  │  Compliance       │    │
│  │  Emotion   │  │  (keyword+LLM)    │    │
│  └─────┬──────┘  └────────┬──────────┘    │
└────────┼──────────────────┼───────────────┘
         ▼                  ▼
┌──────────────┐
│    Fusion    │  Contextual MuRIL + WavLM + trajectory
└──────┬───────┘
       ▼
┌──────────────┐
│   QA Score   │  5-factor weighted grading (A–D)
└──────┬───────┘
       ▼
┌──────────────┐
│  CRM Note    │  TF-IDF extractive summarizer
└──────────────┘
```

---

## Components

| Module | Description |
|---|---|
| `denoising/` | Audio preprocessing: declipping, highpass (80 Hz), notch (50 Hz), compression, spectral Wiener denoising |
| `diarization/` | Speaker diarization via pyannote, voice embedding via SpeechBrain, change detection, confidence scoring |
| `sentiment/stt/` | Google Cloud Speech-to-Text V2 (Chirp 3) with parallel chunk transcription |
| `sentiment/acoustic_emotion/` | Rule-based acoustic emotion: arousal, voice shift, escalation detection |
| `sentiment/emotion_classifier.py` | DistilRoBERTa-based Hindi emotion classifier (fallback) |
| `sentiment/compliance_engine.py` | Keyword fast-path + Gemini Flash LLM verification |
| `sentiment/fusion_layer.py` | Neutral-override fusion, ±0.10 sentiment threshold |
| `sentiment/qa_scorer.py` | 5-factor QA scoring: greeting, product knowledge, process adherence, tone, resolution |
| `sentiment/crm_note_generator.py` | TF-IDF extractive summarizer |
| `svar/` | Next-gen pipeline: ContextualMuRIL, WavLM, fusion_net, temporal_decoder, role inference, calibration |
| `svar/acoustic/` | Speaker-normalized baselines, relative features, trajectory model |
| `svar/models/` | PyTorch models: ContextualMuRIL, WavLMEmotion, FusionNet, TemporalDecoder, RoleClassifier |
| `dashboard/` | Single-page HTML dashboard with 8 tabs, progress tracking, background parallel pipeline |
| `scripts/` | Training and evaluation scripts (`train_muril.py`, `train_wavlm.py`, `evaluate.py`) |
| `configs/` | YAML configs for training and evaluation |

---

## Quick Start

### 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
pip install pyannote.audio speechbrain librosa soundfile numpy scikit-learn joblib google-cloud-speech transformers sentencepiece pyyaml dash
```

### 2. Credentials

Place service account key at `credentials/gcloud-stt.json` (project `sunohq`, requires `roles/speech.admin`).

Place Gemini API keys at `credentials/gemini-api-keys.json`:
```json
["key1", "key2", ...]
```

### 3. Launch Dashboard

```bash
python3 diarization/dashboard_server.py
```

Open http://localhost:8050, select a sample call, click **Analyze**.

### 4. Run Individual Stages

```bash
# Denoising
python -m denoising.pipeline

# Diarization
python -m diarization.pipeline

# Full emotion pipeline
python -m svar.pipeline
```

---

## GPU Notes

- **RTX 2050 (4GB VRAM)**: pyannote runs on GPU, then freed before loading Whisper/emotion models. STT and LLM are cloud-based.
- `torchcodec` is not supported (missing `libnvrtc.so.13`). Audio loading uses `soundfile` + `librosa`.

---

## Emotion Taxonomy

Canonical 16-class Hindi emotion taxonomy:

```
neutral, anticipation, anger, sad, confident, fear, disgusted,
surprised, hopeful, annoyed, compassion, joy, apprehensive,
grateful, guilty, impressed
```

---

## Project Structure

```
SVAR/
├── configs/              YAML configs
├── credentials/          GCloud STT + Gemini keys (.gitignored)
├── dashboard/            Single-page HTML dashboard
├── data/
│   ├── emotions/         EmoInHindi dataset (LREC CSV)
│   └── sample_calls/     Sample audio files
├── denoising/            Audio preprocessing pipeline
├── diarization/          Speaker diarization + dashboard server
├── docs/                 Annotation guidelines
├── scripts/              Training & evaluation
├── sentiment/            Emotion, compliance, QA, CRM
└── svar/                 Next-gen models & pipeline
```

---

## License

MIT
