import numpy as np
from diarization.mfcc_extractor import extract_mfcc
from diarization.prosodic_extractor import extract_prosodic_features
from diarization.lpc_formant_estimator import extract_segment_formants


def extract_speaker_fingerprint(
    audio: np.ndarray,
    sr: int,
    num_ceps: int = 13
) -> np.ndarray:
    """
    Extracts a 32-dimensional L2-normalized voice identity fingerprint vector for an audio segment.

    The 32-dim vector consists of:
      - MFCC statistics (26 dims): 13 coefficient means + 13 coefficient standard deviations.
      - Prosodic metrics (4 dims): pitch mean, pitch std, RMS energy mean, speaking rate (ZCR).
      - LPC Formants (2 dims): F1 formant mean, F2 formant mean.

    Args:
        audio: 1D numpy array of audio samples.
        sr: Sample rate in Hz.
        num_ceps: Number of MFCC coefficients (default: 13).

    Returns:
        32-dimensional float32 numpy array with L2 norm equal to 1.0 (or 0.0 for silent/empty audio).
    """
    if len(audio) == 0 or sr <= 0:
        return np.zeros(32, dtype=np.float32)

    # 1. Spectral features: MFCC mean & std (13 + 13 = 26 dims)
    mfcc_feats = extract_mfcc(audio, sr, num_ceps=num_ceps)
    if len(mfcc_feats) > 0:
        mfcc_means = np.mean(mfcc_feats, axis=0)
        mfcc_stds = np.std(mfcc_feats, axis=0)
    else:
        mfcc_means = np.zeros(num_ceps, dtype=np.float32)
        mfcc_stds = np.zeros(num_ceps, dtype=np.float32)

    # 2. Prosodic features (4 dims)
    prosodic = extract_prosodic_features(audio, sr)
    prosodic_vector = np.array([
        prosodic.get("pitch_mean", 0.0),
        prosodic.get("pitch_std", 0.0),
        prosodic.get("rms_mean", 0.0),
        prosodic.get("speaking_rate_zcr", 0.0)
    ], dtype=np.float32)

    # 3. LPC Formant features (2 dims)
    formants = extract_segment_formants(audio, sr)
    formant_vector = np.array([
        formants.get("f1_mean", 0.0),
        formants.get("f2_mean", 0.0)
    ], dtype=np.float32)

    # 4. Concatenate to 32 dimensions
    raw_vector = np.concatenate([mfcc_means, mfcc_stds, prosodic_vector, formant_vector]).astype(np.float32)

    if len(raw_vector) < 32:
        raw_vector = np.pad(raw_vector, (0, 32 - len(raw_vector)), mode='constant')
    elif len(raw_vector) > 32:
        raw_vector = raw_vector[:32]

    # 5. L2 Normalization
    norm = np.linalg.norm(raw_vector)
    if norm > 1e-10:
        normalized_vector = raw_vector / norm
    else:
        normalized_vector = np.zeros(32, dtype=np.float32)

    return normalized_vector.astype(np.float32)
