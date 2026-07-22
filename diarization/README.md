# SVAR — Speaker Diarization & Baseline Profiling Pipeline

This module implements the speaker diarization, voice activity segmentation, feature extraction, and dynamic baseline profiling components of the SVAR analytics pipeline. It is responsible for segmenting continuous call audio into active speech turns and separating Agent and Customer acoustic profiles.

---

## Pipeline Architecture

The diarization module operates through a sequential multi-stage acoustic processing pipeline:

```
CLEAN AUDIO INPUT
       ↓
┌─────────────────────────────────┐
│ 1. PAUSE SEGMENTER              │ VAD Energy Thresholding &
│    (diarization/pause_segmenter)│ Pause Boundary Splitting (>400ms)
└────────────────┬────────────────┘
                 ↓ Speech Segments
┌─────────────────────────────────┐
│ 2. FEATURE EXTRACTION           │ 
│    - MFCC Extractor (13 coefs)  │ 26 Spectral Stats (Mean + Std)
│    - Prosodic Extractor         │ 4 Pitch ($F_0$), Energy & ZCR Stats
│    - LPC Formant Estimator      │ 2 Formant Resonances ($F_1, F_2$)
└────────────────┬────────────────┘
                 ↓ 32-dim Vector
┌─────────────────────────────────┐
│ 3. SPEAKER FINGERPRINTER        │ $L_2$ Normalization & Voice
│    (speaker_fingerprinter)      │ Embedding Construction
└────────────────┬────────────────┘
                 ↓ Normalized Embedding
┌─────────────────────────────────┐
│ 4. SPEAKER ASSIGNER &           │ Cosine Similarity Matching &
│    DYNAMIC BASELINE TRACKER     │ EWA Profile Updating
└────────────────┬────────────────┘
                 ↓
      AGENT & CUSTOMER TIMELINE
      + TALK DURATION RATIOS
```

---

## Component Specifications

1. **Pause Segmenter ([`pause_segmenter.py`](file:///home/abhinav/Documents/github/SVAR/diarization/pause_segmenter.py)):**
   - Frame-level RMS energy calculation across 25ms windows with 10ms hop.
   - Splits speaker turn boundaries whenever silence gaps exceed **400ms**.
   - Filters brief transient noise bursts under **500ms**.

2. **MFCC Extractor ([`mfcc_extractor.py`](file:///home/abhinav/Documents/github/SVAR/diarization/mfcc_extractor.py)):**
   - Applies pre-emphasis filtering ($\alpha = 0.97$).
   - Computes 26 Mel filterbank log energies and orthonormal DCT-II mapping to 13 cepstral coefficients.

3. **Prosodic Feature Extractor ([`prosodic_extractor.py`](file:///home/abhinav/Documents/github/SVAR/diarization/prosodic_extractor.py)):**
   - **Fundamental Pitch ($F_0$):** Frame autocorrelation in the 50–500 Hz frequency range.
   - **Energy & Rate:** Frame RMS power statistics and Zero-Crossing Rate (ZCR) per second.
   - **Perturbation & Structure:** Jitter, shimmer, pause ratio, crest factor, and peak pitch position (`HiF0`).

4. **LPC Formant Estimator ([`lpc_formant_estimator.py`](file:///home/abhinav/Documents/github/SVAR/diarization/lpc_formant_estimator.py)):**
   - Solves 16th-order Linear Predictive Coding (LPC) polynomial roots in the z-plane.
   - Computes root bandwidths ($BW = -\frac{sr}{\pi} \ln(|z|)$) to isolate valid vocal tract resonances ($F_1, F_2$).

5. **Speaker Fingerprinter ([`speaker_fingerprinter.py`](file:///home/abhinav/Documents/github/SVAR/diarization/speaker_fingerprinter.py)):**
   - Concatenates 26 MFCC stats + 4 prosodic metrics + 2 formant resonances into a unified vector.
   - Applies $L_2$ normalization ($\|v\|_2 = 1.0$) to form a uniform 32-dimensional voice embedding.

6. **Speaker Assigner & Baseline Builder ([`speaker_assigner.py`](file:///home/abhinav/Documents/github/SVAR/diarization/speaker_assigner.py), [`baseline_builder.py`](file:///home/abhinav/Documents/github/SVAR/diarization/baseline_builder.py)):**
   - **Opening Protocol:** Assumes initial call segment is spoken by the **Agent**.
   - **Voice Discovery:** Initializes **Customer** baseline upon detecting a distinct voice fingerprint.
   - **Dynamic EWA Update:** Updates active speaker baselines dynamically ($b_{new} = \text{normalize}((1-\alpha)b_{old} + \alpha \cdot fp)$).
   - **Uncertainty Margin:** Flags segments where cosine similarity differential $|sim_{agent} - sim_{customer}| < 0.05$.

---

## Execution & Benchmarking

Run the complete test suite:
```bash
PYTHONPATH=. venv/bin/python -m unittest discover -s diarization/tests -p "test*.py"
```

Run batch benchmarking across sample call audio files:
```bash
PYTHONPATH=. venv/bin/python diarization/tests/benchmark_diarization.py
```
Benchmark results are exported to [diarization_results.csv](file:///home/abhinav/Documents/github/SVAR/data/sample_calls/diarization_results.csv).
