"""
Baseline-relative acoustic feature construction.

For each turn, computes the z-score deviation from the speaker's
baseline profile, producing a vector that the trajectory model
can interpret as "how different is this turn from the speaker's calm?"
"""
from __future__ import annotations
import numpy as np

from ..schemas import SpeakerBaselineProfile


def build_relative_acoustic_vector(
    prosody_vector: np.ndarray,
    wavlm_embedding: np.ndarray,
    profile: SpeakerBaselineProfile,
) -> np.ndarray:
    """
    Compute baseline-relative acoustic representation for one turn.

    The output is a concatenated vector of:
    - Prosody z-scores (how far each prosodic feature deviates from baseline)
    - Embedding z-scores (how far the WavLM embedding deviates from baseline)
    - Scalar prosody shift (RMS of z-scores, overall acoustic change)
    - Scalar embedding shift (RMS of z-scores, overall voice change)
    - Baseline readiness flag (1.0 if baseline is ready, 0.0 if not)

    Args:
        prosody_vector: (N_feat,) raw prosodic features for this turn.
        wavlm_embedding: (N_emb,) WavLM pooled embedding for this turn.
        profile: Speaker's frozen baseline profile.

    Returns:
        (N_feat + N_emb + 3,) relative feature vector.
        Returns zeros if profile is not ready.
    """
    n_prosody = len(prosody_vector)
    n_emb = len(wavlm_embedding)

    if not profile.ready:
        return np.zeros(n_prosody + n_emb + 3, dtype=np.float32)

    # Z-score prosody features against baseline
    if len(profile.feature_center) == n_prosody and len(profile.feature_scale) == n_prosody:
        prosody_z = np.clip(
            (prosody_vector - profile.feature_center) / profile.feature_scale,
            -6.0, 6.0,
        )
    else:
        prosody_z = np.zeros(n_prosody, dtype=np.float32)

    # Z-score embedding against baseline
    if len(profile.embedding_center) == n_emb and len(profile.embedding_scale) == n_emb:
        embedding_z = np.clip(
            (wavlm_embedding - profile.embedding_center) / profile.embedding_scale,
            -6.0, 6.0,
        )
    else:
        embedding_z = np.zeros(n_emb, dtype=np.float32)

    # Scalar shift scores (RMS of z-scores)
    prosody_shift = float(np.sqrt(np.mean(prosody_z ** 2)))
    embedding_shift = float(np.sqrt(np.mean(embedding_z ** 2)))

    return np.concatenate([
        prosody_z.astype(np.float32),
        embedding_z.astype(np.float32),
        np.array([
            prosody_shift,
            embedding_shift,
            1.0,  # baseline_ready
        ], dtype=np.float32),
    ])


def build_sequence_input(
    turn_vectors: list[np.ndarray],
    durations: list[float],
    quality_scores: list[float],
) -> np.ndarray:
    """
    Build temporal sequence input for the trajectory model.

    For each turn t, the input is:
        [z_prosody, z_embedding, delta_z, duration, quality]

    where delta_z = z_t - z_{t-1} (change from previous turn).

    Args:
        turn_vectors: List of relative acoustic vectors per turn.
        durations: Turn durations in seconds.
        quality_scores: Audio quality scores per turn.

    Returns:
        (T, D) sequence input for the trajectory model.
    """
    if not turn_vectors:
        return np.zeros((0, 0), dtype=np.float32)

    T = len(turn_vectors)
    D = len(turn_vectors[0]) + 1 + 1  # relative + delta_z_mean + duration + quality

    sequence = np.zeros((T, D), dtype=np.float32)

    for t in range(T):
        vec = turn_vectors[t]

        # Delta from previous turn (zero for first turn)
        if t > 0:
            delta = turn_vectors[t] - turn_vectors[t - 1]
        else:
            delta = np.zeros_like(vec)

        # Combine: [relative_vector, delta_mean, duration, quality]
        delta_mean = float(np.sqrt(np.mean(delta ** 2)))
        sequence[t, :len(vec)] = vec
        sequence[t, len(vec)] = delta_mean
        sequence[t, len(vec) + 1] = durations[t]
        # Quality goes into the last slot (shift by 1)
        if len(vec) + 2 < D:
            sequence[t, len(vec) + 2] = quality_scores[t] if t < len(quality_scores) else 0.0

    return sequence
