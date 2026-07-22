import numpy as np
import librosa
from typing import Tuple, Dict


def estimate_formants_lpc(
    frame: np.ndarray,
    sr: int,
    lpc_order: int = 16
) -> Tuple[float, float]:
    """
    Estimates the first two formant frequencies (F1, F2) of an audio frame using LPC.

    Args:
        frame: 1D numpy array of audio samples for a single frame.
        sr: Sample rate in Hz.
        lpc_order: LPC filter order (default: 16).

    Returns:
        Tuple of floats (F1_hz, F2_hz). Returns (0.0, 0.0) if frame is silent or unvoiced.
    """
    if len(frame) == 0 or sr <= 0:
        return (0.0, 0.0)

    # Frame windowing
    windowed = frame * np.hamming(len(frame))
    variance = np.var(windowed)
    if variance < 1e-10:
        return (0.0, 0.0)

    order = min(lpc_order, max(1, len(frame) - 1))

    try:
        # Calculate LPC coefficients
        a = librosa.lpc(windowed, order=order)
        # Solve polynomial roots
        roots = np.roots(a)

        # Filter roots in upper half plane with positive imaginary part
        roots = roots[np.imag(roots) > 0]
        if len(roots) == 0:
            return (0.0, 0.0)

        # Calculate angles and frequencies in Hz
        angles = np.angle(roots)
        freqs = angles * (sr / (2 * np.pi))

        # Calculate bandwidths: BW = -sr / pi * ln(|root|)
        r_abs = np.abs(roots)
        r_abs = np.maximum(r_abs, 1e-10)
        bandwidths = - (sr / np.pi) * np.log(r_abs)

        # Filter valid formant candidates: f in [90 Hz, sr/2 - 90 Hz] and BW < 400 Hz
        valid_mask = (freqs >= 90.0) & (freqs <= (sr / 2.0 - 90.0)) & (bandwidths < 400.0)
        valid_freqs = freqs[valid_mask]

        if len(valid_freqs) == 0:
            valid_freqs = freqs[(freqs >= 90.0) & (freqs <= (sr / 2.0 - 90.0))]

        if len(valid_freqs) == 0:
            return (0.0, 0.0)

        sorted_freqs = np.sort(valid_freqs)
        f1 = float(sorted_freqs[0])
        f2 = float(sorted_freqs[1]) if len(sorted_freqs) > 1 else float(f1 * 2.5)

        return (f1, f2)

    except Exception:
        return (0.0, 0.0)


def extract_segment_formants(
    audio: np.ndarray,
    sr: int,
    frame_size_s: float = 0.03,
    frame_stride_s: float = 0.01,
    lpc_order: int = 16
) -> Dict[str, float]:
    """
    Extracts mean and std of F1 and F2 formants across audio segment frames.

    Args:
        audio: 1D numpy array of audio samples.
        sr: Sample rate in Hz.
        frame_size_s: Frame length in seconds.
        frame_stride_s: Frame stride in seconds.
        lpc_order: LPC polynomial order.

    Returns:
        Dict with keys: 'f1_mean', 'f1_std', 'f2_mean', 'f2_std'.
    """
    if len(audio) == 0 or sr <= 0:
        return {"f1_mean": 0.0, "f1_std": 0.0, "f2_mean": 0.0, "f2_std": 0.0}

    frame_len = int(round(frame_size_s * sr))
    hop_len = int(round(frame_stride_s * sr))

    if len(audio) < frame_len:
        audio = np.pad(audio, (0, frame_len - len(audio)), mode='constant')

    num_frames = max(1, 1 + (len(audio) - frame_len) // hop_len)
    f1_list = []
    f2_list = []

    for i in range(num_frames):
        start = i * hop_len
        end = start + frame_len
        frame = audio[start:end]

        f1, f2 = estimate_formants_lpc(frame, sr, lpc_order=lpc_order)
        if f1 > 0 and f2 > 0:
            f1_list.append(f1)
            f2_list.append(f2)

    if len(f1_list) > 0:
        f1_mean = float(np.mean(f1_list))
        f1_std = float(np.std(f1_list))
        f2_mean = float(np.mean(f2_list))
        f2_std = float(np.std(f2_list))
    else:
        f1_mean = f1_std = f2_mean = f2_std = 0.0

    return {
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "f2_mean": f2_mean,
        "f2_std": f2_std
    }
