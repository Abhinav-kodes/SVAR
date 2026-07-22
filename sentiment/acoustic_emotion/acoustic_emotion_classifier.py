import numpy as np
from typing import Dict, Any, Tuple


EMOTION_PROFILES = {
    "anger": {
        "pitch_d": +2.0,
        "energy_d": +2.0,
        "rate_d": +1.5,
        "jitter_d": +1.0,
        "pause_r": -2.0,
        "hif0": "middle"
    },
    "sadness": {
        "pitch_d": -2.0,
        "energy_d": -2.0,
        "rate_d": -2.0,
        "jitter_d": +0.5,
        "pause_r": +2.0,
        "hif0": "beginning"
    },
    "happiness": {
        "pitch_d": +1.5,
        "energy_d": +1.0,
        "rate_d": +1.0,
        "jitter_d": -0.5,
        "pause_r": -1.0,
        "hif0": "end"
    },
    "fear": {
        "pitch_d": +1.0,
        "energy_d": +0.5,
        "rate_d": +0.5,
        "jitter_d": +2.0,
        "pause_r": +0.5,
        "hif0": "end"
    },
    "neutral": {
        "pitch_d": 0.0,
        "energy_d": 0.0,
        "rate_d": 0.0,
        "jitter_d": -1.0,
        "pause_r": 0.0,
        "hif0": "beginning"
    },
    "stress": {
        "pitch_d": +1.0,
        "energy_d": +1.0,
        "rate_d": +0.5,
        "jitter_d": +1.5,
        "pause_r": -0.5,
        "hif0": "middle"
    },
    "disgust": {
        "pitch_d": -0.5,
        "energy_d": 0.0,
        "rate_d": -0.5,
        "jitter_d": +0.5,
        "pause_r": 0.0,
        "hif0": "beginning"
    }
}


def classify_acoustic_emotion(
    delta_features: Dict[str, float],
    hif0_section: str = "middle",
    confidence_threshold: float = 0.30
) -> Dict[str, Any]:
    """
    Classifies emotion from standardized acoustic feature deltas against research emotion profiles.

    Args:
        delta_features: Dict of feature deltas (pitch_d, energy_d, rate_d, jitter_d, pause_r).
        hif0_section: Section of peak pitch F0 ('beginning', 'middle', 'end').
        confidence_threshold: Threshold below which prediction is marked indeterminate.

    Returns:
        Dict containing:
            - 'emotion': predicted emotion string ('anger', 'sadness', etc.)
            - 'confidence': float score
            - 'indeterminate': bool flag
            - 'all_scores': dict mapping emotion names to normalized probability scores
    """
    feature_keys = ["pitch_d", "energy_d", "rate_d", "jitter_d", "pause_r"]

    query_vec = np.array([float(delta_features.get(k, 0.0)) for k in feature_keys], dtype=np.float32)

    raw_scores = {}

    for emotion, profile in EMOTION_PROFILES.items():
        profile_vec = np.array([float(profile[k]) for k in feature_keys], dtype=np.float32)

        # Distance / Similarity calculation
        dist = np.linalg.norm(query_vec - profile_vec)
        sim = 1.0 / (1.0 + dist)

        # HiF0 position matching bonus
        if profile.get("hif0") == hif0_section:
            sim += 0.30

        raw_scores[emotion] = float(sim)

    # Convert similarity scores to softmax probabilities
    score_vals = np.array(list(raw_scores.values()), dtype=np.float32)
    exp_scores = np.exp(score_vals - np.max(score_vals))
    probs = exp_scores / np.sum(exp_scores)

    all_scores = {emo: float(prob) for emo, prob in zip(raw_scores.keys(), probs)}

    top_emotion = max(all_scores, key=all_scores.get)
    top_confidence = float(all_scores[top_emotion])

    indeterminate = bool(top_confidence < confidence_threshold)

    return {
        "emotion": top_emotion,
        "confidence": round(top_confidence, 4),
        "indeterminate": indeterminate,
        "all_scores": {k: round(v, 4) for k, v in all_scores.items()}
    }
