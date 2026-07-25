"""
Speaker embedding module using SpeechBrain ECAPA-TDNN.

Provides:
1. Centered contextual embeddings for short segments
2. Best-boundary search for merged turn splitting
3. Anchor-based speaker re-classification with raw identity scores
4. Viterbi sequence decoder for ambiguous regions
"""

import math
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple

_classifier = None


def _get_classifier():
    """Lazy-load SpeechBrain ECAPA-TDNN (singleton)."""
    global _classifier
    if _classifier is not None:
        return _classifier

    from speechbrain.inference.speaker import EncoderClassifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )
    return _classifier


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def _encode_array(chunk: np.ndarray) -> np.ndarray:
    """Encode a raw numpy chunk through SpeechBrain."""
    classifier = _get_classifier()
    device = next(classifier.parameters()).device
    chunk_t = torch.tensor(chunk, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        emb = classifier.encode_batch(chunk_t)
    return emb.squeeze().cpu().numpy()


def _embed_chunk(audio: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    """Extract SpeechBrain embedding for a time range."""
    start = max(0, int(start_s * sr))
    end = min(len(audio), int(end_s * sr))
    chunk = audio[start:end]

    min_samples = int(0.3 * sr)
    if len(chunk) < min_samples:
        chunk = np.pad(chunk, (0, min_samples - len(chunk)))

    return _encode_array(chunk)


def _embed_centered(
    audio: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    context_s: float = 1.5,
) -> Optional[np.ndarray]:
    """Embed a segment using a centered contextual window.

    For short segments, extends the window symmetrically to give ECAPA
    enough phonetic material without right-padding zeros.

    Returns None if the window falls in silence.
    """
    mid_s = (start_s + end_s) / 2.0
    half = context_s / 2.0

    lo = max(0, int((mid_s - half) * sr))
    hi = min(len(audio), int((mid_s + half) * sr))
    chunk = audio[lo:hi]

    if np.max(np.abs(chunk)) < 1e-5:
        return None

    return _encode_array(chunk)


def extract_segment_embeddings(
    audio: np.ndarray,
    sr: int,
    segments: List[Dict],
    min_duration_s: float = 0.5,
) -> np.ndarray:
    """Extract SpeechBrain embeddings for each segment.

    For very short segments, pads to min_duration_s before encoding.
    """
    min_samples = int(min_duration_s * sr)
    embeddings = []

    for seg in segments:
        start = int(seg["start_time_s"] * sr)
        end = int(seg["end_time_s"] * sr)
        chunk = audio[start:end]

        if len(chunk) < min_samples:
            chunk = np.pad(chunk, (0, min_samples - len(chunk)))

        emb = _encode_array(chunk)
        embeddings.append(emb)

    return np.array(embeddings, dtype=np.float32)


def _best_split_time(
    audio: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    search_step_s: float = 0.25,
    window_s: float = 1.0,
    drift_threshold: float = 0.30,
) -> Tuple[Optional[float], float]:
    """Search for the best acoustic change-point within a segment.

    Scans the middle 60% of the segment, embedding fixed windows
    immediately before and after each candidate boundary, and returns
    the time that maximizes embedding drift.
    """
    margin = (end_s - start_s) * 0.2
    lo = start_s + margin + window_s
    hi = end_s - margin - window_s

    if lo >= hi:
        mid = (start_s + end_s) / 2.0
        left = _embed_chunk(audio, sr, start_s, mid)
        right = _embed_chunk(audio, sr, mid, end_s)
        d = _cosine_distance(left, right)
        return (mid, d) if d > drift_threshold else (None, d)

    candidates = np.arange(lo, hi + 1e-6, search_step_s)

    best_t, best_d = None, -1.0
    for t in candidates:
        left = _embed_chunk(audio, sr, t - window_s, t)
        right = _embed_chunk(audio, sr, t, t + window_s)
        d = _cosine_distance(left, right)
        if d > best_d:
            best_t, best_d = float(t), d

    return best_t, best_d


def split_merged_turns(
    audio: np.ndarray,
    sr: int,
    segments: List[Dict],
    min_threshold_s: float = 3.5,
    drift_threshold: float = 0.30,
) -> List[Dict]:
    """Detect and split merged turns where pyannote missed a speaker change.

    For segments longer than min_threshold_s, checks start/mid/end embeddings.
    If the embedding drifts significantly, splits at the midpoint.
    """
    result = []
    for seg in segments:
        dur = seg["duration_s"]
        if dur < min_threshold_s:
            result.append(seg)
            continue

        start_s = seg["start_time_s"]
        end_s = seg["end_time_s"]
        mid_s = (start_s + end_s) / 2.0

        emb_start = _embed_chunk(audio, sr, start_s, start_s + 1.5)
        emb_mid = _embed_chunk(audio, sr, mid_s - 0.75, mid_s + 0.75)
        emb_end = _embed_chunk(audio, sr, end_s - 1.5, end_s)

        d_start_mid = _cosine_distance(emb_start, emb_mid)
        d_mid_end = _cosine_distance(emb_mid, emb_end)
        d_start_end = _cosine_distance(emb_start, emb_end)

        if (d_start_mid > drift_threshold or d_mid_end > drift_threshold) and d_start_end > drift_threshold * 0.7:
            split_time = mid_s
            result.append({
                **seg,
                "end_time_s": split_time,
                "duration_s": round(split_time - start_s, 6),
                "end_sample": int(round(split_time * sr)),
                "_needs_reclass": True,
            })
            result.append({
                **seg,
                "start_time_s": split_time,
                "duration_s": round(end_s - split_time, 6),
                "start_sample": int(round(split_time * sr)),
                "_needs_reclass": True,
            })
        else:
            result.append(seg)

    return result


def reassign_speakers(
    audio: np.ndarray,
    sr: int,
    segments: List[Dict],
    num_speakers: int = 2,
) -> List[Dict]:
    """Re-assign speaker labels using SpeechBrain embeddings with anchor-based classification.

    Returns raw identity scores (d_agent, d_customer, sb_margin) for
    downstream sequence decoding.
    """
    if len(segments) < 3:
        return [
            {
                "start_sample": int(round(s["start_time_s"] * sr)),
                "end_sample": int(round(s["end_time_s"] * sr)),
                "start_time_s": s["start_time_s"],
                "end_time_s": s["end_time_s"],
                "duration_s": s["duration_s"],
                "speaker": "spk_0" if i == 0 else "spk_1",
                "sb_d_agent": 0.0,
                "sb_d_customer": 0.0,
                "sb_margin": 0.0,
            }
            for i, s in enumerate(segments)
        ]

    pyannote_groups = {}
    for seg in segments:
        spk = seg.get("pyannote_speaker", "S0")
        if spk not in pyannote_groups:
            pyannote_groups[spk] = []
        pyannote_groups[spk].append(seg)

    if len(pyannote_groups) < 2:
        return [
            {
                "start_sample": int(round(s["start_time_s"] * sr)),
                "end_sample": int(round(s["end_time_s"] * sr)),
                "start_time_s": s["start_time_s"],
                "end_time_s": s["end_time_s"],
                "duration_s": s["duration_s"],
                "speaker": "spk_0",
                "sb_d_agent": 0.0,
                "sb_d_customer": 0.0,
                "sb_margin": 0.0,
            }
            for s in segments
        ]

    spk_list = list(pyannote_groups.keys())
    group_a = pyannote_groups[spk_list[0]]
    group_b = pyannote_groups[spk_list[1]]

    n_anchors = max(3, min(8, min(len(group_a), len(group_b))))

    anchors_a = sorted(group_a, key=lambda s: s["duration_s"], reverse=True)[:n_anchors]
    anchors_b = sorted(group_b, key=lambda s: s["duration_s"], reverse=True)[:n_anchors]

    all_anchors = anchors_a + anchors_b
    anchor_embs = extract_segment_embeddings(audio, sr, all_anchors)
    centroid_a = anchor_embs[:len(anchors_a)].mean(axis=0)
    centroid_b = anchor_embs[len(anchors_a):].mean(axis=0)

    all_embs = extract_segment_embeddings(audio, sr, segments)

    first_pyannote_spk = segments[0].get("pyannote_speaker", "S0")
    if first_pyannote_spk == spk_list[0]:
        centroid_spk0 = centroid_a
        centroid_spk1 = centroid_b
    else:
        centroid_spk0 = centroid_b
        centroid_spk1 = centroid_a

    result = []
    for seg, emb in zip(segments, all_embs):
        d_spk0 = _cosine_distance(emb, centroid_spk0)
        d_spk1 = _cosine_distance(emb, centroid_spk1)
        margin = d_spk1 - d_spk0

        speaker = "spk_0" if margin >= 0 else "spk_1"

        result.append({
            "start_sample": int(round(seg["start_time_s"] * sr)),
            "end_sample": int(round(seg["end_time_s"] * sr)),
            "start_time_s": seg["start_time_s"],
            "end_time_s": seg["end_time_s"],
            "duration_s": seg["duration_s"],
            "speaker": speaker,
            "sb_d_agent": float(d_spk0),
            "sb_d_customer": float(d_spk1),
            "sb_margin": float(margin),
        })

    return result


def decode_region(
    items: List[Dict],
    switch_penalty: float = 0.10,
    stay_penalty: float = 0.08,
) -> List[str]:
    """Viterbi sequence decoder for ambiguous contiguous regions.

    Uses SpeechBrain identity scores as emission probabilities and a light
    conversational transition prior that penalizes same-speaker runs in
    non-overlapping dialogue.

    Parameters
    ----------
    items : list of segment dicts with sb_d_agent, sb_d_customer, start/end times
    switch_penalty : cost for switching speakers (should be small)
    stay_penalty : cost for staying same speaker in non-overlapping dialogue

    Returns
    -------
    list of speaker labels for each item
    """
    if not items:
        return []

    labels = ("spk_0", "spk_1")
    n = len(items)

    emissions = []
    for item in items:
        d_a = item.get("sb_d_agent", 0.5)
        d_c = item.get("sb_d_customer", 0.5)
        total = d_a + d_c
        if total < 1e-8:
            emissions.append({"agent": 0.0, "customer": 0.0})
        else:
            emissions.append({
                "agent": -d_a / max(total, 1e-8),
                "customer": -d_c / max(total, 1e-8),
            })

    dp = [{lab: (-float("inf"), None) for lab in labels} for _ in range(n)]

    for lab in labels:
        dp[0][lab] = (emissions[0][lab], None)

    for i in range(1, n):
        overlaps = items[i]["start_time_s"] < items[i - 1]["end_time_s"] - 0.05

        for current in labels:
            best = (-float("inf"), None)
            for previous in labels:
                score = dp[i - 1][previous][0] + emissions[i][current]

                if not overlaps:
                    if current == previous:
                        score -= stay_penalty
                    else:
                        score -= switch_penalty

                if score > best[0]:
                    best = (score, previous)

            dp[i][current] = best

    last = max(labels, key=lambda lab: dp[-1][lab][0])
    out = [last]
    for i in range(n - 1, 0, -1):
        out.append(dp[i][out[-1]][1])
    return list(reversed(out))
