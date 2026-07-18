# Call Analytics Pipeline — Updated Day-by-Day Roadmap
*Incorporating EmoInHindi Dataset (44,247 utterances, IIT Patna LREC 2022)*

---

## Project Architecture

```
RAW CALL AUDIO
      ↓
┌─────────────────────────┐
│  PART 1: DENOISING      │  Days 1–5
│  + ENHANCEMENT PIPELINE │
└──────────┬──────────────┘
           ↓ clean audio
┌─────────────────────────┐
│  PART 2: DIARIZATION    │  Days 6–11
│  + BASELINE PROFILING   │
└──────┬──────────┬───────┘
       ↓          ↓
  Agent        Customer
  segments     segments
       ↓          ↓
┌─────────────────────────┐
│  PART 3: SENTIMENT      │  Days 12–19
│  ANALYSIS (Voice+Text)  │
└──────────┬──────────────┘
           ↓
   Per-speaker emotion
   timeline + QA scores
           ↓
┌─────────────────────────┐
│  INTEGRATION + POLISH   │  Days 20–25
│  FastAPI + Celery +     │
│  RabbitMQ + MongoDB     │
└─────────────────────────┘
```

---

## Part 1 — Denoising + Enhancement Pipeline (Days 1–5)

### Day 1 — Project Setup + Audio Loader + Quality Metrics

**Goal:** Project scaffold + all audio quality measurement components working

**Setup:**
```
call-analytics/
├── part1_denoising/
│   ├── audio_loader.py
│   ├── audio_enhancer.py
│   ├── snr_calculator.py
│   ├── clipping_detector.py
│   ├── silence_ratio.py
│   └── tests/
│       └── enhancement_test.py
├── part2_diarization/
├── part3_sentiment/
│   ├── data/
│   │   └── emoin_hindi.csv      ← your dataset goes here
│   └── preprocessing/
├── backend/
└── data/sample_calls/           ← download 10 files from Kaggle Hindi Call Center
```

**Tasks:**
- Download 10 sample Hindi call recordings from Kaggle Hindi Call Center dataset
- Implement `audio_loader.py` — load WAV/MP3, align & merge agent and customer channels if separate, resample to 16kHz mono via `scipy.signal.resample`
- Implement `snr_calculator.py` — SNR using numpy FFT (signal power vs noise floor from first 0.5s)
- Implement `clipping_detector.py` — samples at ±max amplitude threshold → clipping ratio

**Deliverable:** `quality_report.json` per test file showing SNR + clipping ratio

---

### Day 2 — Silence Ratio + Basic VAD

**Goal:** Measure silence and detect speech regions

**Tasks:**
- Implement `silence_ratio.py` — RMS energy per 25ms frame, frames below threshold = silence
- Implement `vad_basic.py` — energy-threshold VAD, returns boolean speech/silence array per frame
- Unit tests for all Day 1 + Day 2 components using synthetic signals (known SNR, known clipping)
- Test on 3 real sample calls, verify output makes sense visually

**Deliverable:** Speech/silence timeline plot for a sample call

---

### Day 3 — Audio Enhancement Pipeline (Filters + Compression)

**Goal:** Remove hum, rumble, and normalize loudness before denoising

**Tasks:**
- Implement `highpass_filter.py` — Butterworth high-pass at 80 Hz (remove rumble)
- Implement `notch_filter.py` — IIR notch at 50 Hz (Indian mains hum)
- Implement `compressor.py` — RMS-based dynamic range compression (threshold -20 dB, ratio 3:1)
- Implement `declipper.py` — simple interpolation for clipped samples (optional)
- Create `enhancement_pipeline.py` chaining: high-pass → notch → compression → de-clipper
- Test on 3 sample calls: verify 50 Hz spike removed from FFT, quiet speech boosted

**Deliverable:** `enhance_audio()` function returning processed audio, FFT plot showing 50 Hz notch

---

### Day 4 — Spectral Wiener Denoiser

**Goal:** Core denoising — runs after enhancement pipeline (the most mathematically intensive component of Part 1)

**Tasks:**
- Implement `spectral_denoiser.py` to run on already-enhanced audio (output from Day 3):
  1. STFT — frame signal into 25ms windows, apply Hamming window, FFT per frame
  2. Estimate noise PSD from first 0.5s (assumed silence/noise-only region)
  3. Wiener gain per frequency bin:
     ```
     PSD_noise  = mean(|X_noise(f)|²)
     PSD_signal = max(|X(f)|² - PSD_noise, 0)
     gain       = PSD_signal / (PSD_signal + PSD_noise + ε)
     X_clean(f) = gain * X(f)
     ```
  4. iSTFT — overlap-add reconstruction back to time domain
- Listen to before/after audio on 3 sample calls

**Deliverable:** Working denoiser that reduces background noise audibly

---

### Day 5 — SNR Delta Measurement + Part 1 Integration

**Goal:** Measure denoiser effectiveness + wrap Part 1 into pipeline class

**Tasks:**
- Post-denoise SNR calculation and delta computation
- Measure 50 Hz power before/after notch filter (should drop by 20+ dB)
- Create `DenoiserPipeline` class running all steps, returning:
  ```json
  {
    "snr_before_db": 8.3,
    "snr_after_db": 16.7,
    "snr_improvement_db": 8.4,
    "clipping_ratio": 0.002,
    "silence_ratio": 0.31,
    "hum_removed": true,
    "compression_applied": true,
    "audio_quality_grade": "PASS"
  }
  ```
- Benchmark on all 10 sample calls → save results to CSV
- Write `README_part1.md` with SNR improvement numbers as evidence

**Deliverable:** `DenoiserPipeline` class + benchmark CSV showing before/after SNR across 10 calls

---

## Part 2 — Diarization + Baseline Profiling (Days 6–11)

### Day 6 — MFCC Extractor Integration

**Goal:** Integrate the core feature extractor used throughout the diarization and sentiment modules.

**Tasks:**
- Validate output of MFCC extractor on enhanced audio — verify lower jitter/shimmer noise floor vs raw audio
- Implement `mfcc_extractor.py` leveraging standard libraries (such as `librosa`):
  1. Pre-emphasis filtering
  2. Overlapping frame extraction
  3. Windowing (e.g. Hamming)
  4. Mel filterbank conversion (Slaney normalization)
  5. Log energy calculation
  6. DCT-II mapping to 13 MFCC coefficients
- Validate output against standard `librosa.feature.mfcc()` and ensure alignment
- Unit test confirming output shape `(n_frames, 13)`

**Deliverable:** Validated MFCC extractor matching standard librosa output

---

### Day 7 — Prosodic Feature Extractor

**Goal:** Extract pitch, energy, speaking rate, jitter, shimmer, HiF0 position per segment

**Tasks:**
- Implement `prosodic_extractor.py`:
  - **Pitch (F0):** Autocorrelation — find lag of max autocorrelation in 50–500Hz range per frame
  - **RMS Energy:** Per-frame root-mean-square amplitude
  - **Speaking Rate:** Zero-crossing rate per second as phoneme rate proxy
  - **Jitter:** Mean absolute diff between consecutive F0 values / mean F0
  - **Shimmer:** Mean absolute diff between consecutive RMS values / mean RMS
  - **Pause Ratio:** Silence frames / total frames (from VAD)
  - **Compression Ratio:** peak/RMS ratio post-compression (helps detect forced speech)
  - **HiF0 Position:** Peak F0 index / total frames → "beginning" / "middle" / "end"
- Return as flat numpy vector (32-dim)

**Deliverable:** `extract_prosodic_features(segment, sr)` returning validated feature vector

---

### Day 8 — Pause Segmenter + Speaker Fingerprinter

**Goal:** Split audio at turn boundaries, create per-segment voice identity vectors

**Tasks:**
- Implement `pause_segmenter.py`:
  - Use VAD output, group consecutive speech frames into segments
  - Split at pauses > 400ms (configurable threshold)
  - Return list of `(start_sample, end_sample, audio_chunk)`
- Implement `speaker_fingerprinter.py`:
  - MFCC stats: mean + std of 13 coefficients = 26 values
  - Append: F0 mean, F0 std, RMS energy, speaking rate, F1 mean, F2 mean
  - L2-normalize → 32-dim fingerprint vector

**Deliverable:** Segmented call with fingerprint vector per segment

---

### Day 9 — Speaker Assigner + Baseline Builder

**Goal:** Label each segment as Agent or Customer

**Tasks:**
- Implement `speaker_assigner.py`:
  - First segment = Speaker 1 baseline, first clearly different-sounding = Speaker 2 baseline
  - Per new segment: cosine similarity to both baselines → assign to closest
  - `cosine_similarity(a, b)` using standard numpy vector operations
  - "Uncertain" flag when similarity difference < 0.05
- Implement `baseline_builder.py`:
  - Store baseline fingerprint per speaker
  - Rolling EWA update: `baseline = 0.9 * baseline + 0.1 * new_fp`
  - First speaker to talk in call = Agent (greeting convention)
- Test on 5 calls — manually verify agent/customer separation

**Deliverable:** Diarization with agent/customer labels on test calls

---

### Day 10 — LPC Formant Estimator

**Goal:** Add F1/F2 formant features for better speaker-identity separation

**Tasks:**
- Implement `lpc_formant_estimator.py`:
  - Linear Predictive Coding (LPC) using library tools (e.g. `librosa.lpc`)
  - Find roots of LPC polynomial using standard solver (`numpy.roots`)
  - Roots near unit circle in upper half-plane → formant frequencies
  - Return F1, F2 (first two formants in Hz)
- Integrate into speaker fingerprinter
- Re-test diarization accuracy on 5 calls with improved fingerprint

**Deliverable:** Improved fingerprinter with formant features, re-tested on 5 calls

---

### Day 11 — Part 2 Integration + Testing

**Goal:** `DiarizationPipeline` class, end-to-end test with Part 1 output

**Tasks:**
- Create `DiarizationPipeline` (consuming enhanced + denoised audio from Part 1):
  ```python
  result = pipeline.process(clean_audio)
  # Returns:
  # {
  #   "agent":    {"baseline_fp": [...], "segments": [...]},
  #   "customer": {"baseline_fp": [...], "segments": [...]}
  # }
  ```
- Run Part 1 → Part 2 chain on 10 test calls end-to-end
- Log talk ratio (agent vs customer speaking time) per call to CSV
- Write `README_part2.md`

**🚩 Minimum Viable Checkpoint:** Days 1–11 alone = shippable project. If deadline is close, push this to GitHub now.

**Deliverable:** `DiarizationPipeline` end-to-end on 10 calls

---

## Part 3 — Sentiment Analysis (Days 12–19)

### Day 12 — EmoInHindi Dataset Preprocessing

**Goal:** Clean, map, and prepare the EmoInHindi dataset for MuRIL fine-tuning

**Tasks:**
- Load `emoin_hindi.csv` (44,247 rows)
- Parse multi-label `emotions` and `emoIntensity` columns:
  ```python
  df["emotion_list"]   = df["emotions"].str.split(",")
  df["intensity_list"] = df["emoIntensity"].str.split(",").apply(
      lambda x: [int(i) for i in x]
  )
  ```
- Apply 16 → 6 emotion mapping with intensity-based primary label selection:
  ```python
  LABEL_MAP = {
      "anger":"anger",    "annoyed":"anger",
      "sad":"sadness",    "guilty":"sadness",
      "fear":"fear",      "apprehensive":"fear",
      "joy":"happiness",  "grateful":"happiness",
      "impressed":"happiness", "compassion":"happiness",
      "disgusted":"disgust",
      "neutral":"neutral", "confident":"neutral",
      "anticipation":"neutral", "hopeful":"neutral", "surprised":"neutral",
  }
  ```
- Cap neutral at 10,000, keep all minority classes
- Add 2-utterance context window per sample (prepend prev 2 utterances)
- **Split by dialogueId** (not by row!) — 80/10/10 train/val/test
- Save processed splits to `train.csv`, `val.csv`, `test.csv`

**Expected class distribution after processing:**
```
neutral    10,000
anger       9,200
happiness   8,300
fear        7,000
sadness     6,400
disgust     4,000
Total      44,900
```

**Deliverable:** 3 clean CSV files ready for MuRIL training

---

### Day 13 — Delta Normalizer + Acoustic Emotion Classifier

**Goal:** Build the mathematical core of the acoustic emotion classification

**Tasks:**
- Implement `delta_normalizer.py`:
  - Z-score normalize: `delta = (current - baseline) / (baseline_std + ε)`
  - Output normalized delta vector in range [-1, +1] per feature
- Build `EMOTION_PROFILES` dictionary from Yildirim 2004 + Cao 2014 + PLOS ONE 2025:
  ```python
  EMOTION_PROFILES = {
    "anger":     {"pitch_d":+2.0,"energy_d":+2.0,"rate_d":+1.5,
                  "jitter":+1.0,"pause_r":-2.0,"hif0":"middle"},
    "sadness":   {"pitch_d":-2.0,"energy_d":-2.0,"rate_d":-2.0,
                  "jitter":+0.5,"pause_r":+2.0,"hif0":"beginning"},
    "happiness": {"pitch_d":+1.5,"energy_d":+1.0,"rate_d":+1.0,
                  "jitter":-0.5,"pause_r":-1.0,"hif0":"end"},
    "fear":      {"pitch_d":+1.0,"energy_d":+0.5,"rate_d":+0.5,
                  "jitter":+2.0,"pause_r":+0.5,"hif0":"end"},
    "neutral":   {"pitch_d": 0.0,"energy_d": 0.0,"rate_d": 0.0,
                  "jitter":-1.0,"pause_r": 0.0,"hif0":"beginning"},
    "stress":    {"pitch_d":+1.0,"energy_d":+1.0,"rate_d":+0.5,
                  "jitter":+1.5,"pause_r":-0.5,"hif0":"middle"},
    "disgust":   {"pitch_d":-0.5,"energy_d": 0.0,"rate_d":-0.5,
                  "jitter":+0.5,"pause_r": 0.0,"hif0":"beginning"},
  }
  ```
- Implement `acoustic_emotion_classifier.py`:
  - Cosine similarity vs each profile
  - +0.3 HiF0 position bonus for matching profile
  - Return top emotion + all confidence scores
  - Flag "indeterminate" if top confidence < 0.3

**Deliverable:** `classify_emotion(delta_features) → {emotion, confidence, all_scores}`

---

### Day 14 — Whisper STT Integration

**Goal:** Transcribe each speaker segment to text

**Tasks:**
- Download `ARTPARK-IISc/whisper-large-v3-vaani-hindi` from HuggingFace
- Implement `stt_transcriber.py` using HuggingFace pipeline
- Set `language="hi"` forced decoder for Hindi
- Batch-process all segments from Part 2 output
- Fallback to `openai/whisper-base` if VRAM insufficient for large model
- Add transcript per segment to the data structure

**Deliverable:** All 10 test call segments with Hindi transcripts attached

---

### Day 15 — MuRIL Dataset Loader + Model Architecture

**Goal:** Build the PyTorch dataset + multi-head MuRIL model class

**Tasks:**
- Implement `EmotionDataset` (PyTorch Dataset class):
  ```python
  class EmotionDataset(Dataset):
      def __init__(self, df, tokenizer, max_len=128):
          self.texts  = df["input_text"].tolist()    # context + utterance
          self.labels = df["label"].map(LABEL2ID).tolist()
          self.tokenizer = tokenizer
          self.max_len = max_len

      def __getitem__(self, idx):
          enc = self.tokenizer(
              self.texts[idx],
              truncation=True, padding="max_length",
              max_length=self.max_len, return_tensors="pt"
          )
          return {
              "input_ids":      enc["input_ids"].squeeze(),
              "attention_mask": enc["attention_mask"].squeeze(),
              "label":          torch.tensor(self.labels[idx])
          }
  ```
- Implement `MultiTaskMuRIL` with 3 heads:
  ```python
  class MultiTaskMuRIL(nn.Module):
      def __init__(self):
          super().__init__()
          self.encoder = AutoModel.from_pretrained("google/muril-base-cased")
          self.emotion_head   = nn.Linear(768, 6)   # EmoInHindi
          self.sentiment_head = nn.Linear(768, 3)   # IndicSentiment (pos/neu/neg)
          self.intent_head    = nn.Linear(768, 6)   # BFSI intents

      def forward(self, input_ids, attention_mask):
          out = self.encoder(input_ids, attention_mask).pooler_output
          return {
              "emotion":   self.emotion_head(out),
              "sentiment": self.sentiment_head(out),
              "intent":    self.intent_head(out),
          }
  ```
- Compute inverse-frequency class weights for emotion head
- Verify model forward pass works on a single batch

**Deliverable:** `EmotionDataset` + `MultiTaskMuRIL` with confirmed forward pass

---

### Day 16 — MuRIL Training Run

**Goal:** Fine-tune on EmoInHindi (emotion head) + IndicSentiment (sentiment head)

**Tasks:**
- Training loop with:
  - AdamW optimizer, lr=2e-5, weight decay=0.01
  - Linear warmup scheduler (10% of steps)
  - Gradient checkpointing if VRAM tight on RTX 2050
  - Batch size 16, 5 epochs
  - Weighted CrossEntropyLoss for emotion head
  - Save best checkpoint by validation emotion macro-F1
- Log per-epoch: train loss, val loss, val accuracy, val macro-F1
- Start training before sleep — ~45–60 min on RTX 2050

**Expected results:**
```
Epoch 1: val_macro_f1 ≈ 0.55
Epoch 3: val_macro_f1 ≈ 0.68
Epoch 5: val_macro_f1 ≈ 0.72–0.74
```

**Per-class expected F1 (from EmoInHindi paper baseline of 66.24%):**
```
anger:     ~0.80  (most data, most distinct)
sadness:   ~0.74
fear:      ~0.72
happiness: ~0.70
neutral:   ~0.78
disgust:   ~0.62  (least data)
```

**Deliverable:** Saved `muril_emotion_best.pt` checkpoint + training curves CSV

---

### Day 17 — Fusion Layer + Compliance Engine

**Goal:** Combine acoustic + text branches + build compliance rule engine

**Tasks:**
- Implement `fusion_layer.py` with confidence-gated fusion:
  ```python
  def fuse(text_emotion, text_conf, acoustic_emotion, acoustic_conf):
      if text_conf > 0.70:
          return text_emotion           # text is certain → trust it
      elif acoustic_conf > 0.65:
          return acoustic_emotion       # acoustic is certain
      else:
          # neither confident → weighted blend
          scores = 0.55 * text_scores + 0.45 * acoustic_scores
          return argmax(scores)
  ```
- Implement `compliance_engine.py` from scratch:
  - RBI/IRDAI violation regex patterns (Hindi + English)
  - Abusive language keyword list
  - Levenshtein distance (DP, from scratch):
    ```python
    def levenshtein(s1, s2):
        dp = [[0]*(len(s2)+1) for _ in range(len(s1)+1)]
        for i in range(len(s1)+1): dp[i][0] = i
        for j in range(len(s2)+1): dp[0][j] = j
        for i in range(1,len(s1)+1):
            for j in range(1,len(s2)+1):
                dp[i][j] = dp[i-1][j-1] if s1[i-1]==s2[j-1] else                             1+min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[-1][-1]
    ```
  - Fuzzy match: flag if Levenshtein ≤ 2 against violation keywords

**Deliverable:** `FusionLayer` + `ComplianceEngine` classes tested on sample transcripts

---

### Day 18 — QA Scoring Formula + CRM Note Generator

**Goal:** Compute per-agent quality score + auto-generate CRM notes

**Tasks:**
- Implement `qa_scorer.py` with configurable weighted formula:
  ```python
  score = (
      0.30 * avg_customer_sentiment_score    +  # 0–1, higher = happier customer
      0.25 * (1 - compliance_flag_ratio)     +  # 0 flags = perfect
      0.20 * agent_sentiment_stability       +  # low variance in agent emotion
      0.15 * intent_resolution_rate          +  # did intent get resolved?
      0.10 * talk_ratio_score                   # agent:customer ~50:50 is ideal
  )
  # score → 0–100, grade: A(85+) B(70+) C(55+) D(<55)
  ```
  - Weights in `config.yaml` (easily tunable)
- Implement `crm_note_generator.py` using Gemini 1.5 Flash free API:
  - Input: transcript + emotion timeline + intent + compliance flags
  - Prompt: "Generate a 3-sentence professional CRM note from this Hindi call transcript..."
  - Fallback: TF-IDF extractive summarizer (no API needed)

**Deliverable:** QA score + CRM note generated for each of 10 test calls

---

### Day 19 — Part 3 Integration + End-to-End Test

**Goal:** Wire all Part 3 components into `SentimentPipeline`, run full chain

**Tasks:**
- Create `SentimentPipeline` class consuming Part 2 output
- Run full Part 1 → Part 2 → Part 3 chain on all 10 test calls
- Verify emotion timelines look reasonable per call
- Log all outputs to MongoDB (local Docker instance)
- Write `README_part3.md` with methodology, citing all 4 papers:
  - Schewski et al. PLOS ONE 2025
  - Yildirim et al. INTERSPEECH 2004
  - Cao et al. Speech Prosody 2014
  - Singh et al. (EmoInHindi) LREC 2022

**Deliverable:** Full 3-part chain end-to-end, outputs in MongoDB

---

## Integration + Polish (Days 20–25)

### Day 20 — FastAPI Backend

**Goal:** Expose the pipeline as a production-ready REST API

**Routes:**
```
POST /calls/upload           → accept audio file, return job_id
GET  /calls/{job_id}/status  → poll for completion
GET  /calls/{job_id}/results → full analysis JSON
GET  /dashboard/agent/{id}   → agent QA summary (last 30 calls)
GET  /dashboard/overview     → all agents ranked by score
```

**Tasks:**
- FastAPI async endpoints + background tasks for pipeline execution
- JWT authentication middleware (simple HS256 token)
- Pydantic response models for all endpoints
- `/health` endpoint for Docker health checks

**Deliverable:** FastAPI server running locally with all routes returning valid responses

---

### Day 21 — Celery + RabbitMQ Queue

**Goal:** Decouple API from pipeline with async task queue

**Tasks:**
- Set up RabbitMQ via Docker: `docker run -d rabbitmq:3-management`
- Implement `publisher.py` — push audio job to queue on upload
- Implement `consumer.py` — Celery worker picks up job, runs pipeline
- Chain tasks: `denoise.s() | diarize.s() | analyze_sentiment.s()`
- Test: upload audio → appears in RabbitMQ queue → worker processes → result in MongoDB

**Deliverable:** Async pipeline fully functional via message queue

---

### Day 22 — MongoDB Schema + Redis Cache

**Goal:** Persist all results, cache dashboard queries

**Tasks:**
- MongoDB document schemas:
  ```
  Call:    {call_id, timestamp, duration, agent_id, qa_score, grade}
  Segment: {call_id, speaker, start, end, transcript, emotion, confidence}
  Agent:   {agent_id, avg_score, call_count, emotion_distribution}
  ```
- Save full analysis per call to MongoDB
- Redis cache for `GET /dashboard/agent/{id}` — TTL 5 minutes
- Rolling 30-call window for agent QA score tracker

**Deliverable:** Full persistence layer with Redis caching working

---

### Day 23 — Docker Compose Setup

**Goal:** Single-command full system deployment

**`docker-compose.yml` services:**
```yaml
services:
  api:       FastAPI container (port 8000)
  worker:    Celery worker container
  rabbitmq:  Message broker (port 5672, management 15672)
  mongodb:   Database (port 27017)
  redis:     Cache (port 6379)
```

**Tasks:**
- `Dockerfile` for API + worker (shared base image)
- `docker-compose.yml` with proper networking + health checks
- Environment variables via `.env` file (API keys, DB connection strings)
- Test: `docker-compose up` → full system live
- Verify: `docker-compose up --scale worker=3` works for horizontal scaling

**Deliverable:** `docker-compose up` starts entire system from cold

---

### Day 24 — README + Architecture Diagram

**Goal:** Professional GitHub README that tells the whole story

**README Sections:**
1. Project title + one-line description
2. Architecture diagram (ASCII block diagram)
3. Part-by-part description with "built from scratch" highlights:
   - High-pass Butterworth filter
   - IIR notch filter
   - RMS compressor
   - Wiener denoiser
   - MFCC extractor
   - LPC formant estimator
4. Audio Enhancement section — 50 Hz notch filter, highpass rumble removal, dynamic compression (built from scratch)
   - Benchmark table showing 50 Hz hum reduction: 20+ dB
4. Dataset section — EmoInHindi citation + Kaggle Hindi Call Center + Vaani/IISc
5. Research papers cited (all 4, with links)
6. Setup: `git clone → docker-compose up`
7. Sample output JSON (real output from your test calls)
8. Benchmark table (SNR improvement, diarization accuracy, emotion F1)
9. Known limitations (acted vs spontaneous speech, fear/anxiety acoustic detection)
10. Future work (multilingual, real-time streaming, zkML sentiment proofs)

**Deliverable:** `README.md` that makes a hiring manager want to run the project

---

### Day 25 — Final Testing + Demo Video

**Goal:** End-to-end validation, demo recording, GitHub push

**Tasks:**
- Run 20 calls through full pipeline — zero crashes required
- Record 3-minute demo:
  - 00:00–00:45 — raw noisy audio → denoised output (audible difference)
  - 00:45–01:30 — diarization separating agent/customer timeline
  - 01:30–02:15 — emotion timeline visualization per speaker
  - 02:15–03:00 — dashboard API response showing QA score + CRM note
- Push to GitHub with clean commit history
- Link demo video in README

**Deliverable:** Public GitHub repo live with demo video

---

## Summary

| Part | Days | Hardest Day | Key Components |
|---|---|---|---|
| Denoising + Enhancement | 1–5 | Day 4 (Wiener filter) | SNR calculator, NOTCH filter, high-pass, compressor, Wiener denoiser, VAD |
| Diarization | 6–11 | Day 10 (Formants) | MFCC, formants, cosine fingerprinter |
| Sentiment | 12–19 | Day 16 (MuRIL training) | EmoInHindi preprocessing, V-A classifier, Levenshtein |
| Integration | 20–25 | Day 21 (queue architecture) | FastAPI, Celery, Docker Compose |
| **Total** | **25 days** | | **Custom integration of standard DSP & ML tools** |

---

## Dataset Citations (Use in README + Resume)

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

## Daily Time Budget (3–4 hrs/day)

| Block | Activity |
|---|---|
| First 30 min | Review previous day, fix leftover issues |
| Next 2 hrs | Main implementation task |
| Next 45 min | Unit tests + validation on real audio |
| Last 15 min | Git commit + one-line note on what was learned |

---

## Minimum Viable Checkpoint — Day 11

Days 1–11 = fully working denoising + diarization pipeline with benchmarks.
This alone outperforms the average intern-level project in India.
If your application deadline hits before Day 19, push at Day 11, label Part 3 as "in progress."