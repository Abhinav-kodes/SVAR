"""
Low-level acoustic feature extraction for baseline and trajectory models.

Extracts per-turn features that are speaker-normalized via z-score
against the speaker's baseline profile.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


# ── Feature names for reference ──
FEATURE_NAMES = [
    "log_energy_mean", "log_energy_std",
    "f0_median", "f0_iqr", "f0_slope",
    "voiced_ratio",
    "speech_rate_proxy",
    "pause_ratio",
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_slope_mean",
    "hnr_mean",
    "jitter", "shimmer",
]
N_LOW_LEVEL_FEATURES = len(FEATURE_NAMES)


def extract_baseline_features(
    audio: np.ndarray,
    sr: int = 16000,
) -> np.ndarray:
    """
    Extract low-level acoustic features from a mono audio segment.

    Args:
        audio: (n_samples,) float32 mono audio at sr Hz.
        sr: Sample rate.

    Returns:
        (N_LOW_LEVEL_FEATURES,) feature vector.
    """
    try:
        import librosa
    except ImportError:
        return np.zeros(N_LOW_LEVEL_FEATURES, dtype=np.float32)

    if len(audio) < sr * 0.1:
        return np.zeros(N_LOW_LEVEL_FEATURES, dtype=np.float32)

    features = []

    # Energy
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    log_energy = np.log(rms + 1e-8)
    features.append(np.mean(log_energy))
    features.append(np.std(log_energy))

    # F0 (pitch)
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=60, fmax=500, sr=sr,
        frame_length=2048, hop_length=512,
    )
    f0_valid = f0[~np.isnan(f0)]
    if len(f0_valid) > 2:
        features.append(np.median(f0_valid))
        q75, q25 = np.percentile(f0_valid, [75, 25])
        features.append(q75 - q25)  # IQR
        # Slope: linear trend over time
        t = np.arange(len(f0_valid))
        if len(t) > 1:
            slope = np.polyfit(t, f0_valid, 1)[0]
            features.append(slope / (np.median(f0_valid) + 1e-8))  # normalized
        else:
            features.append(0.0)
    else:
        features.extend([0.0, 0.0, 0.0])

    # Voiced ratio
    if voiced_flag is not None:
        features.append(np.mean(voiced_flag.astype(float)))
    else:
        features.append(0.0)

    # Speech rate proxy (onset density)
    onsets = librosa.onset.onset_detect(y=audio, sr=sr, hop_length=512)
    duration_s = len(audio) / sr
    features.append(len(onsets) / max(duration_s, 0.01))

    # Pause ratio (energy below threshold)
    threshold = np.mean(rms) * 0.1 if len(rms) > 0 else 0
    pause_frames = np.sum(rms < threshold) / max(len(rms), 1)
    features.append(pause_frames)

    # Spectral centroid
    sc = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features.append(np.mean(sc) / (sr / 2))  # normalized
    features.append(np.std(sc) / (sr / 2))

    # Spectral slope (roll-off ratio)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0]
    features.append(np.mean(rolloff) / (sr / 2))

    # Harmonics-to-noise ratio (via autocorrelation)
    hnr = _compute_hnr(audio, sr)
    features.append(hnr)

    # Jitter and shimmer (only if audio is clean enough)
    jitter, shimmer = _compute_jitter_shimmer(audio, sr)
    features.append(jitter)
    features.append(shimmer)

    result = np.array(features, dtype=np.float32)

    # Pad/trim to expected size
    if len(result) < N_LOW_LEVEL_FEATURES:
        result = np.pad(result, (0, N_LOW_LEVEL_FEATURES - len(result)))
    elif len(result) > N_LOW_LEVEL_FEATURES:
        result = result[:N_LOW_LEVEL_FEATURES]

    return result


def _compute_hnr(audio: np.ndarray, sr: int) -> float:
    """Estimate harmonics-to-noise ratio via autocorrelation."""
    try:
        import librosa
        autocorr = np.correlate(audio, audio, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / (autocorr[0] + 1e-8)

        # Find first peak after lag 0
        min_lag = int(sr * 0.003)  # 3ms minimum pitch period
        max_lag = int(sr * 0.025)  # 25ms maximum pitch period
        if max_lag > len(autocorr):
            return 0.0

        search = autocorr[min_lag:max_lag]
        if len(search) == 0:
            return 0.0

        peak_idx = np.argmax(search) + min_lag
        peak_val = autocorr[peak_idx]

        if peak_val > 0:
            hnr_db = 10 * np.log10(peak_val / (1 - peak_val + 1e-8))
            return np.clip(hnr_db / 30.0, -1.0, 1.0)  # normalized
        return 0.0
    except Exception:
        return 0.0


def _compute_jitter_shimmer(audio: np.ndarray, sr: int) -> Tuple[float, float]:
    """Estimate jitter (pitch perturbation) and shimmer (amplitude perturbation)."""
    try:
        import librosa
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=60, fmax=500, sr=sr,
            frame_length=2048, hop_length=512,
        )
        f0_valid = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
        f0_valid = f0_valid[~np.isnan(f0_valid)]

        if len(f0_valid) < 3:
            return 0.0, 0.0

        # Jitter: mean absolute pitch difference / mean pitch
        diffs = np.abs(np.diff(f0_valid))
        jitter = np.mean(diffs) / (np.mean(f0_valid) + 1e-8)

        # Shimmer: mean absolute amplitude difference / mean amplitude
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        if len(rms) > 1:
            shimmer = np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-8)
        else:
            shimmer = 0.0

        return float(np.clip(jitter, 0, 1)), float(np.clip(shimmer, 0, 1))
    except Exception:
        return 0.0, 0.0
