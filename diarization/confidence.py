"""
Embedding-based confidence scoring and separability detection.

Replaces duration-only heuristics with metrics computed from pyannote's
internal speaker embeddings. Uses silhouette scoring for per-segment
confidence and rolling centroid distance for separability detection.
"""

import numpy as np
from typing import List, Dict, Optional


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def _extract_segment_embeddings(
    chunk_embeddings: np.ndarray,
    segmentation_data: np.ndarray,
    segmentation_window,
    segments: List[Dict],
) -> np.ndarray:
    """Extract an embedding for each segment from chunk embeddings.

    Finds the chunk whose center is closest to each segment's midpoint,
    then selects the powerset class with highest activation at that frame.
    """
    chunk_start = segmentation_window.start
    chunk_step = segmentation_window.step
    chunk_duration = segmentation_window.duration
    num_chunks = chunk_embeddings.shape[0]
    num_frames = segmentation_data.shape[1]
    dim = chunk_embeddings.shape[2]

    result = np.zeros((len(segments), dim), dtype=np.float32)

    for i, seg in enumerate(segments):
        seg_center = (seg["start_time_s"] + seg["end_time_s"]) / 2.0

        chunk_idx = int(round((seg_center - chunk_start) / chunk_step))
        chunk_idx = max(0, min(chunk_idx, num_chunks - 1))

        chunk_time_start = chunk_start + chunk_idx * chunk_step
        relative_time = seg_center - chunk_time_start
        frame_idx = int(relative_time / chunk_duration * num_frames)
        frame_idx = max(0, min(frame_idx, num_frames - 1))

        class_probs = segmentation_data[chunk_idx, frame_idx]
        active_class = int(np.argmax(class_probs))
        result[i] = chunk_embeddings[chunk_idx, active_class]

    return result


def _centroid_distance_fallback(
    seg_embeddings: np.ndarray,
    numeric_labels: np.ndarray,
    valid_mask: np.ndarray,
    segments: List[Dict],
) -> List[Dict]:
    """Fallback: confidence = nearest_other_dist / (own_dist + nearest_other_dist)."""
    valid_indices = np.where(valid_mask)[0]
    num_speakers = int(numeric_labels[valid_mask].max()) + 1
    dim = seg_embeddings.shape[1]

    centroids = np.zeros((num_speakers, dim), dtype=np.float32)
    for s in range(num_speakers):
        mask = valid_mask & (numeric_labels == s)
        if mask.sum() > 0:
            centroids[s] = seg_embeddings[mask].mean(axis=0)

    for idx in valid_indices:
        emb = seg_embeddings[idx]
        label = numeric_labels[idx]
        own_dist = _cosine_distance(emb, centroids[label])

        other_dists = [
            _cosine_distance(emb, centroids[j])
            for j in range(num_speakers) if j != label
        ]
        nearest_other = min(other_dists) if other_dists else own_dist

        total = own_dist + nearest_other
        confidence = nearest_other / total if total > 0 else 0.5
        confidence = float(np.clip(confidence, 0.0, 1.0))
        segments[idx]["confidence"] = round(confidence, 4)
        segments[idx]["uncertain"] = confidence < 0.5

    for i, seg in enumerate(segments):
        if not valid_mask[i]:
            seg["confidence"] = 0.5
            seg["uncertain"] = True

    return segments


def compute_segment_confidence(
    chunk_embeddings: np.ndarray,
    segmentation_data: np.ndarray,
    segmentation_window,
    segments: List[Dict],
    speaker_labels: List[str],
) -> List[Dict]:
    """Compute embedding-based silhouette confidence for each segment.

    Uses scikit-learn's silhouette_samples to measure how well-separated
    each segment's embedding is from other segments of different speakers.

    Parameters
    ----------
    chunk_embeddings : (num_chunks, num_classes, dim) from pyannote hook
    segmentation_data : (num_chunks, num_frames, num_classes) from pyannote hook
    segmentation_window : SlidingWindow with start, duration, step
    segments : pipeline output segments (modified in-place)
    speaker_labels : e.g. ['SPEAKER_00', 'SPEAKER_01']

    Returns
    -------
    segments with 'confidence' (float 0-1) and 'uncertain' (bool) added
    """
    from sklearn.metrics import silhouette_samples

    if len(segments) < 3 or len(speaker_labels) < 2:
        for seg in segments:
            seg["confidence"] = 0.5
            seg["uncertain"] = True
        return segments

    seg_embeddings = _extract_segment_embeddings(
        chunk_embeddings, segmentation_data, segmentation_window, segments
    )

    label_to_int = {label: i for i, label in enumerate(speaker_labels)}
    numeric_labels = np.array([
        label_to_int.get(seg["speaker"], -1) for seg in segments
    ])

    valid_mask = numeric_labels >= 0
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 3:
        for seg in segments:
            seg["confidence"] = 0.5
            seg["uncertain"] = True
        return segments

    valid_embeddings = seg_embeddings[valid_mask]
    valid_labels = numeric_labels[valid_mask]

    unique_labels, counts = np.unique(valid_labels, return_counts=True)
    if np.any(counts < 2):
        return _centroid_distance_fallback(
            seg_embeddings, numeric_labels, valid_mask, segments
        )

    sil_scores = silhouette_samples(valid_embeddings, valid_labels, metric="cosine")

    for idx, sil_score in zip(valid_indices, sil_scores):
        confidence = float(np.clip((sil_score + 1.0) / 2.0, 0.0, 1.0))
        segments[idx]["confidence"] = round(confidence, 4)
        segments[idx]["uncertain"] = confidence < 0.5

    for i, seg in enumerate(segments):
        if not valid_mask[i]:
            seg["confidence"] = 0.5
            seg["uncertain"] = True

    return segments


def compute_rolling_separability(
    chunk_embeddings: np.ndarray,
    segmentation_window,
    global_centroids: np.ndarray,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
    total_duration_s: Optional[float] = None,
) -> List[Dict]:
    """Compute rolling cosine distance between speaker centroids.

    For each sliding window, chunks are assigned to the nearest global
    centroid, then the two group centroids are compared. Low distance
    means speakers are acoustically similar in that region.
    """
    if global_centroids is None or global_centroids.shape[0] < 2:
        return []

    chunk_start = segmentation_window.start
    chunk_step = segmentation_window.step
    chunk_duration = segmentation_window.duration
    num_chunks = chunk_embeddings.shape[0]
    num_speakers = global_centroids.shape[0]

    if total_duration_s is None:
        total_duration_s = (
            chunk_start + (num_chunks - 1) * chunk_step + chunk_duration
        )

    chunk_centers = np.array([
        chunk_start + i * chunk_step + chunk_duration / 2.0
        for i in range(num_chunks)
    ])

    dim = global_centroids.shape[1]
    active_embs = np.zeros((num_chunks, dim), dtype=np.float32)
    assignments = np.zeros(num_chunks, dtype=int)

    for i in range(num_chunks):
        embs = chunk_embeddings[i]
        norms = np.linalg.norm(embs, axis=1)
        active_embs[i] = embs[np.argmax(norms)]
        dists = np.array([
            _cosine_distance(active_embs[i], c) for c in global_centroids
        ])
        assignments[i] = int(np.argmin(dists))

    curve = []
    pos = chunk_start
    while pos + window_sec <= total_duration_s + step_sec:
        window_end = pos + window_sec
        in_window = (chunk_centers >= pos) & (chunk_centers < window_end)

        effective_min = 1 if pos < chunk_start + window_sec else 2
        if in_window.sum() < effective_min:
            pos += step_sec
            continue

        group_centroids = []
        for s in range(num_speakers):
            mask = in_window & (assignments == s)
            if mask.sum() > 0:
                group_centroids.append(active_embs[mask].mean(axis=0))

        if len(group_centroids) >= 2:
            dist = _cosine_distance(group_centroids[0], group_centroids[1])
            curve.append({
                "start_s": round(pos, 2),
                "end_s": round(min(window_end, total_duration_s), 2),
                "separability": round(dist, 4),
            })

        pos += step_sec

    return curve


def detect_low_separability_regions(
    separability_curve: List[Dict],
    threshold_percentile: float = 25.0,
    pad_s: float = 2.0,
) -> List[Dict]:
    """Identify contiguous time regions where separability is below threshold."""
    if not separability_curve:
        return []

    scores = [p["separability"] for p in separability_curve]
    threshold = float(np.percentile(scores, threshold_percentile))

    low_windows = [p for p in separability_curve if p["separability"] <= threshold]
    if not low_windows:
        return []

    regions = []
    current = {
        "start_s": low_windows[0]["start_s"],
        "end_s": low_windows[0]["end_s"],
        "min_separability": low_windows[0]["separability"],
    }

    for w in low_windows[1:]:
        if w["start_s"] <= current["end_s"] + 1.0:
            current["end_s"] = max(current["end_s"], w["end_s"])
            current["min_separability"] = min(
                current["min_separability"], w["separability"]
            )
        else:
            regions.append(current)
            current = {
                "start_s": w["start_s"],
                "end_s": w["end_s"],
                "min_separability": w["separability"],
            }
    regions.append(current)

    for r in regions:
        r["start_s"] = round(max(0.0, r["start_s"] - pad_s), 2)
        r["end_s"] = round(r["end_s"] + pad_s, 2)

    return regions
