import numpy as np
from typing import Dict, Any, Tuple


def extract_pitch_autocorrelation(
    frame: np.ndarray,
    sr: int,
    f_min: float = 50.0,
    f_max: float = 500.0,
    voicing_threshold: float = 0.3
) -> float:
    """
    Estimates the fundamental frequency (F0/pitch) of an audio frame using autocorrelation.

    Args:
        frame: 1D numpy array of audio samples for a single frame.
        sr: Sampling rate in Hz.
        f_min: Minimum frequency bound in Hz (default: 50.0 Hz).
        f_max: Maximum frequency bound in Hz (default: 500.0 Hz).
        voicing_threshold: Minimum normalized autocorrelation peak to declare a frame voiced.

    Returns:
        Fundamental frequency F0 in Hz (0.0 if unvoiced or frame too short).
    """
    n_samples = len(frame)
    if n_samples == 0:
        return 0.0

    # Subtract frame mean
    frame_centered = frame - np.mean(frame)
    variance = np.var(frame_centered)
    if variance < 1e-10:
        return 0.0

    # Calculate lag range bounds
    lag_min = int(np.floor(sr / f_max))
    lag_max = int(np.ceil(sr / f_min))

    if lag_max >= n_samples:
        lag_max = n_samples - 1

    if lag_min >= lag_max or lag_min <= 0:
        return 0.0

    # FFT-based autocorrelation for fast computation
    n_fft = 1 << (2 * n_samples - 1).bit_length()
    fft_frame = np.fft.rfft(frame_centered, n=n_fft)
    autocorr = np.fft.irfft(fft_frame * np.conj(fft_frame), n=n_fft)[:n_samples]

    if autocorr[0] <= 1e-10:
        return 0.0

    # Normalize autocorrelation
    autocorr_norm = autocorr / autocorr[0]

    # Search for peak within frequency lag limits
    search_region = autocorr_norm[lag_min:lag_max + 1]
    if len(search_region) == 0:
        return 0.0

    peak_idx = np.argmax(search_region)
    peak_val = search_region[peak_idx]

    if peak_val >= voicing_threshold:
        best_lag = lag_min + peak_idx
        pitch_hz = float(sr / best_lag)
        return pitch_hz

    return 0.0


def extract_prosodic_features(
    audio: np.ndarray,
    sr: int,
    frame_size_s: float = 0.03,
    frame_stride_s: float = 0.01,
    voicing_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Extracts comprehensive prosodic features (pitch, energy, speaking rate, jitter,
    shimmer, pause ratio, compression ratio, HiF0 position) from an audio segment.

    Args:
        audio: 1D numpy array of audio samples.
        sr: Sample rate in Hz.
        frame_size_s: Frame duration in seconds (default: 30ms).
        frame_stride_s: Hop length in seconds (default: 10ms).
        voicing_threshold: Autocorrelation peak threshold for voiced detection.

    Returns:
        Dictionary containing extracted feature metrics and a 32-dimensional numpy vector 'vector'.
    """
    if len(audio) == 0 or sr <= 0:
        empty_vector = np.zeros(32, dtype=np.float32)
        return {
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "pitch_min": 0.0,
            "pitch_max": 0.0,
            "rms_mean": 0.0,
            "rms_std": 0.0,
            "rms_min": 0.0,
            "rms_max": 0.0,
            "speaking_rate_zcr": 0.0,
            "jitter": 0.0,
            "shimmer": 0.0,
            "pause_ratio": 0.0,
            "compression_ratio": 0.0,
            "hif0_relative_pos": 0.0,
            "hif0_section": "beginning",
            "voiced_ratio": 0.0,
            "vector": empty_vector
        }

    frame_length = int(round(frame_size_s * sr))
    hop_length = int(round(frame_stride_s * sr))

    if len(audio) < frame_length:
        # Pad short audio to at least one frame length
        audio = np.pad(audio, (0, frame_length - len(audio)), mode='constant')

    # Extract overlapping frames
    num_frames = max(1, 1 + (len(audio) - frame_length) // hop_length)
    
    pitches = np.zeros(num_frames, dtype=np.float32)
    rms_energies = np.zeros(num_frames, dtype=np.float32)
    zcrs = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]

        # 1. Pitch
        pitches[i] = extract_pitch_autocorrelation(
            frame, sr, f_min=50.0, f_max=500.0, voicing_threshold=voicing_threshold
        )

        # 2. RMS Energy
        rms_energies[i] = np.sqrt(np.mean(frame**2))

        # 3. Zero-Crossing Rate per second
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(frame))))
        zcrs[i] = (zero_crossings / frame_size_s)

    # Voiced frame metrics
    voiced_mask = pitches > 0
    voiced_pitches = pitches[voiced_mask]
    voiced_count = len(voiced_pitches)
    voiced_ratio = float(voiced_count / num_frames)

    if voiced_count > 0:
        pitch_mean = float(np.mean(voiced_pitches))
        pitch_std = float(np.std(voiced_pitches))
        pitch_min = float(np.min(voiced_pitches))
        pitch_max = float(np.max(voiced_pitches))
        pitch_p25 = float(np.percentile(voiced_pitches, 25))
        pitch_p75 = float(np.percentile(voiced_pitches, 75))
    else:
        pitch_mean = pitch_std = pitch_min = pitch_max = pitch_p25 = pitch_p75 = 0.0

    # RMS statistics
    rms_mean = float(np.mean(rms_energies))
    rms_std = float(np.std(rms_energies))
    rms_min = float(np.min(rms_energies))
    rms_max = float(np.max(rms_energies))
    rms_p25 = float(np.percentile(rms_energies, 25))
    rms_p75 = float(np.percentile(rms_energies, 75))

    # Speaking Rate (mean ZCR)
    speaking_rate_zcr = float(np.mean(zcrs))

    # Jitter: Mean absolute difference between consecutive voiced pitch periods / mean period
    if voiced_count >= 2:
        pitch_periods = 1.0 / voiced_pitches
        period_diffs = np.abs(np.diff(pitch_periods))
        mean_period = np.mean(pitch_periods)
        jitter = float(np.mean(period_diffs) / mean_period) if mean_period > 0 else 0.0
    else:
        jitter = 0.0

    # Shimmer: Mean absolute difference between consecutive RMS values / mean RMS
    if num_frames >= 2 and rms_mean > 1e-10:
        rms_diffs = np.abs(np.diff(rms_energies))
        shimmer = float(np.mean(rms_diffs) / rms_mean)
    else:
        shimmer = 0.0

    # Pause ratio: fraction of frames below dynamic silence threshold
    silence_threshold = max(1e-4, 0.05 * rms_max)
    silent_frames = np.sum(rms_energies < silence_threshold)
    pause_ratio = float(silent_frames / num_frames)

    # Compression ratio (Crest Factor: Peak / RMS)
    peak_val = float(np.max(np.abs(audio)))
    compression_ratio = float(peak_val / (rms_mean + 1e-10))

    # HiF0 Position (Peak F0 relative location)
    if voiced_count > 0:
        peak_f0_idx = int(np.argmax(pitches))
        hif0_relative_pos = float(peak_f0_idx / num_frames)
    else:
        peak_f0_idx = 0
        hif0_relative_pos = 0.0

    if hif0_relative_pos < 0.33:
        hif0_section = "beginning"
    elif hif0_relative_pos < 0.66:
        hif0_section = "middle"
    else:
        hif0_section = "end"

    # Construct flat 32-dimensional feature vector
    vector = np.array([
        pitch_mean,             # 0
        pitch_std,              # 1
        pitch_min,              # 2
        pitch_max,              # 3
        pitch_p25,              # 4
        pitch_p75,              # 5
        voiced_ratio,           # 6
        rms_mean,               # 7
        rms_std,                # 8
        rms_min,                # 9
        rms_max,                # 10
        rms_p25,                # 11
        rms_p75,                # 12
        speaking_rate_zcr,      # 13
        float(np.std(zcrs)),    # 14
        float(np.max(zcrs)),    # 15
        jitter,                 # 16
        shimmer,                # 17
        pause_ratio,            # 18
        compression_ratio,      # 19
        hif0_relative_pos,      # 20
        1.0 if hif0_section == "beginning" else 0.0, # 21
        1.0 if hif0_section == "middle" else 0.0,    # 22
        1.0 if hif0_section == "end" else 0.0,       # 23
        float(num_frames),      # 24
        float(len(audio) / sr), # 25
        float(voiced_count),    # 26
        float(peak_f0_idx),     # 27
        peak_val,               # 28
        0.0,                    # 29 (reserved)
        0.0,                    # 30 (reserved)
        0.0                     # 31 (reserved)
    ], dtype=np.float32)

    return {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "rms_min": rms_min,
        "rms_max": rms_max,
        "speaking_rate_zcr": speaking_rate_zcr,
        "jitter": jitter,
        "shimmer": shimmer,
        "pause_ratio": pause_ratio,
        "compression_ratio": compression_ratio,
        "hif0_relative_pos": hif0_relative_pos,
        "hif0_section": hif0_section,
        "voiced_ratio": voiced_ratio,
        "vector": vector
    }
