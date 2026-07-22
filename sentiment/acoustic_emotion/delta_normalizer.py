import numpy as np
from typing import Dict, Any


def compute_feature_deltas(
    current_features: Dict[str, float],
    baseline_features: Dict[str, float],
    baseline_stds: Dict[str, float] = None,
    epsilon: float = 1e-5
) -> Dict[str, float]:
    """
    Computes standardized Z-score deviation deltas for prosodic features relative to speaker baseline:
        delta = (current - baseline) / (baseline_std + epsilon)

    Args:
        current_features: Dictionary of acoustic metrics for current segment.
        baseline_features: Dictionary of baseline acoustic metrics for speaker.
        baseline_stds: Optional dictionary of baseline standard deviations for metrics.
        epsilon: Small numerical tolerance to prevent zero-division.

    Returns:
        Dictionary of standardized feature deltas:
            - pitch_d
            - energy_d
            - rate_d
            - jitter_d
            - pause_r
    """
    if baseline_stds is None:
        baseline_stds = {}

    keys_map = {
        "pitch_mean": "pitch_d",
        "rms_mean": "energy_d",
        "speaking_rate_zcr": "rate_d",
        "jitter": "jitter_d",
        "pause_ratio": "pause_r"
    }

    deltas = {}
    for feature_key, delta_key in keys_map.items():
        curr_val = float(current_features.get(feature_key, 0.0))
        base_val = float(baseline_features.get(feature_key, curr_val))
        base_std = float(baseline_stds.get(feature_key, 1.0))
        if base_std <= 1e-5:
            base_std = 1.0

        # Calculate Z-score delta
        delta = (curr_val - base_val) / (base_std + epsilon)
        # Clip to safe numerical bounds [-5.0, +5.0]
        deltas[delta_key] = float(np.clip(delta, -5.0, 5.0))

    return deltas
