# SVAR — Audio Denoising and Enhancement Pipeline

This module implements the audio preprocessing, enhancement, and denoising stages of the SVAR roadmap. It is responsible for ingestion, preprocessing, signal-level enhancement, and spectral noise reduction of noisy customer support calls.

---

## Pipeline Architecture

The pipeline consists of a sequential chain of signal processing modules:

1.  **Audio Loading & Resampling:** Ingests raw formats (`.wav`, `.mp3`, `.opus`), downmixes to mono, and standardises to 16 kHz.
2.  **Cubic Spline De-clipper:** Interpolates samples exceeding $\geq 0.99$ amplitude threshold using cubic splines fitted on adjacent unclipped regions.
3.  **High-Pass Filter:** 5th-order Butterworth high-pass filter at 80 Hz to eliminate low-frequency rumble, table thumps, and wind noise.
4.  **Notch Filter:** IIR notch filter at 50 Hz to remove India's mains AC electrical hum.
5.  **Dynamic Range Compressor:** RMS-based dynamic range compressor (threshold -20 dBFS, ratio 3:1) with smoothed exponential attack/release times to standardise talker volumes.
6.  **Spectral Wiener Denoiser:** Short-Time Fourier Transform (STFT) Wiener filter estimating the noise Power Spectral Density (PSD) from the initial 0.5 seconds of silence and attenuating noise bins with a spectral floor to prevent musical noise artifacts.

---

## Verification & Benchmark Results

We benchmarked the pipeline on all raw sample calls. The results prove a substantial improvement in signal-to-noise ratio (SNR) across noisy audio files while leaving already clean files untouched.

### Summary Table

| Filename | Input SNR (dB) | Output SNR (dB) | SNR Improvement (dB) | Silence Ratio | Grade |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `sample_audio.mp3` | 54.09 | 53.91 | -0.18 | 12.64% | PASS |
| `sample_audio_2.opus` | 38.02 | 47.13 | **+9.11** | 16.92% | PASS |
| `sample_audio_3.opus` | 33.97 | 38.71 | **+4.73** | 14.83% | PASS |

*Note: For the already clean `sample_audio.mp3`, the Wiener denoiser correctly bypassed noise reduction to avoid introducing artifacts (difference is negligible at -0.18 dB).*

Detailed results are exported to [benchmark_results.csv](file:///home/abhinav/Documents/github/SVAR/data/sample_calls/benchmark_results.csv).

---

## Component Files

*   `audio_loader.py` — Raw audio loading, resampling, and initial SNR estimation.
*   `clipping_detector.py` — Ratio of clipped samples.
*   `silence_ratio.py` — RMS energy VAD and silence percentage.
*   `highpass_filter.py` — Butterworth 80 Hz high-pass filter.
*   `notch_filter.py` — IIR 50 Hz notch filter.
*   `compressor.py` — Dynamic range compressor.
*   `declipper.py` — Cubic spline peak interpolator.
*   `spectral_denoiser.py` — Wiener noise filter (STFT/iSTFT).
*   `pipeline.py` — Master `DenoiserPipeline` class.
*   `tests/benchmark.py` — Batch benchmark execution script.
*   `tests/test_synthetic.py` — Automated synthetic test suite.

---

## Running the Pipeline & Benchmarks

To run the benchmarking script and generate the output files and CSV report:
```bash
PYTHONPATH=. venv/bin/python denoising/tests/benchmark.py
```
To run the synthetic unit tests verifying all mathematical modules:
```bash
venv/bin/python -m unittest discover -s denoising/tests -p "test_synthetic.py"
```
