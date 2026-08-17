# SVAR — Speaker Diarization Pipeline

Speaker diarization module using pyannote.audio for segmentation and SpeechBrain for embeddings. Separates agent/customer voices, computes confidence scores, and detects speaker changes.

---

## Pipeline

```
Denoised Audio
      │
      ▼
┌─────────────────────┐
│ pyannote            │  speaker-diarization-3.1
│ segmentation        │  GPU-accelerated, lazy-loaded singleton
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ SpeechBrain         │  ECAPA-TDNN embeddings
│ speaker_embedder    │  anchor-based re-classification
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ change_detector     │  Merge false splits, detect single-speaker
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ confidence          │  Silhouette scoring + rolling separability
└────────┬────────────┘
         ▼
  Speaker Segments
  + Confidence Scores
```

---

## Components

| File | Description |
|---|---|
| `pipeline.py` | `DiarizationPipeline` — orchestrates pyannote → embedding → change detection → confidence |
| `speaker_embedder.py` | SpeechBrain ECAPA-TDNN: centered embeddings, best-boundary search, anchor re-classification, Viterbi decoding |
| `change_detector.py` | ML-based false-split merger + single-speaker detection (trained `change_detector_model.joblib`) |
| `confidence.py` | Silhouette-based per-segment confidence, rolling centroid separability curve |
| `prosodic_extractor.py` | Pitch (F0), energy, ZCR, jitter, shimmer, pause ratio — used by acoustic emotion pipeline |

---

## GPU Management

pyannote runs on GPU when available. The pipeline explicitly frees GPU memory after diarization completes (`del _MODEL` + `gc.collect` + `torch.cuda.empty_cache()`) before downstream stages load their own models.

---

## Running

```bash
# Diarization only
python -m diarization.pipeline

# Tests
PYTHONPATH=. venv/bin/python -m unittest discover -s diarization/tests -p "test*.py"
```

---

## Output Format

Each segment includes:
- `speaker`: `spk_0`, `spk_1`, etc.
- `start_s`, `end_s`, `duration_s`
- `confidence`: 0.0–1.0 (silhouette-based)
- `separability`: speaker-pair separability score
- `uncertain`: True if below threshold
