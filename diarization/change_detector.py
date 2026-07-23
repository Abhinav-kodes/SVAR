"""
Speaker-change detector using pyannote chunk embeddings and segmentation patterns.

Detects false splits (same-speaker boundaries incorrectly created by pyannote)
and merges them, plus detects single-speaker audio.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "change_detector_model.joblib",
)

_model_cache = None


def _load_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not os.path.exists(_MODEL_PATH):
        return None
    import joblib
    _model_cache = joblib.load(_MODEL_PATH)
    return _model_cache


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.dot(a, b))
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    return d / n if n > 1e-8 else 0.0


def _extract_features(
    seg_data: np.ndarray,
    embeddings: np.ndarray,
    chunk_idx: int,
) -> Optional[np.ndarray]:
    n_slots = embeddings.shape[1]
    if chunk_idx < 1 or chunk_idx >= embeddings.shape[0]:
        return None

    prev_emb = embeddings[chunk_idx - 1]
    curr_emb = embeddings[chunk_idx]

    features = []
    sims = []
    best_sim = -1.0
    best_l2 = float("inf")
    for i in range(n_slots):
        for j in range(n_slots):
            s = _cosine_sim(prev_emb[i], curr_emb[j])
            sims.append(s)
            if s > best_sim:
                best_sim = s
                best_l2 = float(np.linalg.norm(prev_emb[i] - curr_emb[j]))

    sims_arr = np.array(sims)
    features.extend([
        best_sim,
        float(sims_arr.min()),
        float(sims_arr.max()),
        float(sims_arr.mean()),
        best_l2,
    ])
    features.append(float(np.mean(np.abs(prev_emb - curr_emb))))

    if seg_data is not None and chunk_idx < seg_data.shape[0]:
        prev_seg = seg_data[chunk_idx - 1]
        curr_seg = seg_data[chunk_idx]
        prev_active = np.argmax(prev_seg, axis=1)
        curr_active = np.argmax(curr_seg, axis=1)
        change_ratio = float(np.mean(prev_active != curr_active))
        features.append(change_ratio)
        n_frames = seg_data.shape[1]
        features.append(
            abs(float(curr_seg[:, 1].sum() - prev_seg[:, 1].sum())) / max(n_frames, 1)
        )
        features.append(
            abs(float(curr_seg[:, 2].sum() - prev_seg[:, 2].sum())) / max(n_frames, 1)
        )
    else:
        features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


def score_boundaries(
    segments: List[Dict],
    seg_data: np.ndarray,
    embeddings: np.ndarray,
) -> List[float]:
    """Score each inter-segment boundary for being a real speaker change.
    
    Returns a list of scores (higher = more likely a real change).
    Length = len(segments) - 1.
    """
    model = _load_model()
    if model is None:
        return [0.5] * (len(segments) - 1)

    clf = model["clf"]
    scaler = model["scaler"]
    threshold = model["threshold"]

    scores = []
    for i in range(1, len(segments)):
        chunk_idx = i
        if chunk_idx >= embeddings.shape[0]:
            scores.append(threshold)
            continue

        feat = _extract_features(seg_data, embeddings, chunk_idx)
        if feat is None:
            scores.append(threshold)
            continue

        feat_scaled = scaler.transform(feat.reshape(1, -1))
        prob = clf.predict_proba(feat_scaled)[0, 1]
        scores.append(float(prob))

    return scores


def merge_false_splits(
    segments: List[Dict],
    seg_data: np.ndarray,
    embeddings: np.ndarray,
    threshold: Optional[float] = None,
) -> List[Dict]:
    """Remove false speaker-change boundaries by merging adjacent segments
    with the SAME speaker label where the change detector confidence is below threshold.
    
    This only merges over-split segments within a single speaker's turn.
    Segments with different speaker labels are never merged (safe).
    
    Returns merged segment list with updated durations.
    """
    model = _load_model()
    if model is None or len(segments) < 3:
        return segments

    if threshold is None:
        threshold = model["threshold"]

    scores = score_boundaries(segments, seg_data, embeddings)

    merged = [segments[0].copy()]
    merges = 0
    for i in range(1, len(segments)):
        score = scores[i - 1]
        same_speaker = segments[i]["speaker"] == merged[-1]["speaker"]

        if score >= threshold or not same_speaker:
            merged.append(segments[i].copy())
        else:
            prev = merged[-1]
            prev["end_time_s"] = segments[i]["end_time_s"]
            prev["end_sample"] = segments[i].get("end_sample", segments[i]["end_time_s"])
            prev["duration_s"] = prev["end_time_s"] - prev["start_time_s"]
            merges += 1

    return merged


def detect_single_speaker(
    segments: List[Dict],
    seg_data: np.ndarray,
    embeddings: np.ndarray,
) -> Tuple[bool, float]:
    """Detect if audio contains a single speaker.
    
    Uses the change detector: if ALL boundaries score below threshold,
    it's likely single-speaker.
    
    Returns (is_single_speaker, mean_score).
    """
    model = _load_model()
    if model is None or len(segments) < 3:
        return False, 0.5

    scores = score_boundaries(segments, seg_data, embeddings)
    threshold = model["threshold"]
    
    n_above = sum(1 for s in scores if s >= threshold)
    mean_score = float(np.mean(scores))
    
    is_single = n_above == 0 and mean_score < threshold
    
    return is_single, mean_score
