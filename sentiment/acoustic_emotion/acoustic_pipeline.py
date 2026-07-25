"""
Acoustic emotion analysis pipeline.

Wires diarization output → prosodic extraction → delta normalization →
emotion classification for each speaker segment.

Requires:
  - diarization.prosodic_extractor (extract_prosodic_features)
  - sentiment.acoustic_emotion.delta_normalizer (compute_feature_deltas)
  - sentiment.acoustic_emotion.acoustic_emotion_classifier (classify_acoustic_emotion)
"""

import numpy as np
from typing import Dict, Any, List, Optional

from diarization.prosodic_extractor import extract_prosodic_features
from sentiment.acoustic_emotion.delta_normalizer import compute_feature_deltas
from sentiment.acoustic_emotion.acoustic_emotion_classifier import classify_acoustic_emotion


def _build_speaker_baseline() -> Dict[str, Any]:
    """Returns an empty baseline accumulator for a speaker."""
    return {
        "features": [],
        "mean": None,
        "std": None,
        "count": 0,
    }


def _update_baseline(baseline: Dict[str, Any], prosodic: Dict[str, Any]) -> None:
    """Accumulates prosodic features into the running baseline."""
    baseline["features"].append(prosodic)
    baseline["count"] += 1

    feature_keys = ["pitch_mean", "rms_mean", "speaking_rate_zcr", "jitter", "pause_ratio"]
    means = {}
    stds = {}
    for key in feature_keys:
        values = [f[key] for f in baseline["features"]]
        means[key] = float(np.mean(values))
        stds[key] = float(np.std(values)) if len(values) > 1 else 1.0

    baseline["mean"] = means
    baseline["std"] = stds


def analyze_acoustic_emotions(
    audio: np.ndarray,
    sr: int,
    segments: List[Dict[str, Any]],
    min_segments_for_baseline: int = 3,
) -> List[Dict[str, Any]]:
    """Run acoustic emotion analysis on diarized segments.

    For each segment:
      1. Extract prosodic features (pitch, energy, jitter, etc.)
      2. Compute Z-score deltas against the speaker's rolling baseline
      3. Classify emotion from delta features

    The baseline is built incrementally: the first `min_segments_for_baseline`
    segments for each speaker establish the baseline, after which deltas are
    computed against it.

    Parameters
    ----------
    audio : full call audio array
    sr : sample rate
    segments : list of dicts from DiarizationPipeline with at minimum
        start_time_s, end_time_s, speaker keys
    min_segments_for_baseline : how many segments before deltas are meaningful

    Returns
    -------
    segments list with acoustic_emotion dict added to each entry
    """
    baselines: Dict[str, Dict[str, Any]] = {}

    for seg in segments:
        speaker = seg.get("speaker", "spk_0")
        start_s = seg.get("start_time_s", 0.0)
        end_s = seg.get("end_time_s", 0.0)

        start_sample = max(0, int(start_s * sr))
        end_sample = min(len(audio), int(end_s * sr))
        segment_audio = audio[start_sample:end_sample]

        if len(segment_audio) < int(0.3 * sr):
            seg["acoustic_emotion"] = {
                "emotion": "neutral",
                "confidence": 0.0,
                "indeterminate": True,
                "all_scores": {},
                "prosodic_features": {},
                "deltas": {},
            }
            continue

        prosodic = extract_prosodic_features(segment_audio, sr)

        if speaker not in baselines:
            baselines[speaker] = _build_speaker_baseline()

        bl = baselines[speaker]

        if bl["count"] < min_segments_for_baseline:
            _update_baseline(bl, prosodic)
            seg["acoustic_emotion"] = {
                "emotion": "neutral",
                "confidence": 0.0,
                "indeterminate": True,
                "all_scores": {},
                "prosodic_features": {k: v for k, v in prosodic.items() if k != "vector"},
                "deltas": {},
            }
            continue

        deltas = compute_feature_deltas(prosodic, bl["mean"], bl["std"])

        emotion_result = classify_acoustic_emotion(
            deltas, hif0_section=prosodic.get("hif0_section", "middle")
        )

        _update_baseline(bl, prosodic)

        seg["acoustic_emotion"] = {
            "emotion": emotion_result["emotion"],
            "confidence": emotion_result["confidence"],
            "indeterminate": emotion_result["indeterminate"],
            "all_scores": emotion_result["all_scores"],
            "prosodic_features": {k: v for k, v in prosodic.items() if k != "vector"},
            "deltas": deltas,
        }

    return segments
