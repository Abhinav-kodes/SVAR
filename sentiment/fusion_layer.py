import numpy as np
from typing import Dict, Any, Optional, List


TEXT_CONF_THRESHOLD = 0.70
ACOUSTIC_CONF_THRESHOLD = 0.65
TEXT_WEIGHT = 0.55
ACOUSTIC_WEIGHT = 0.45

EMOTION_TO_SENTIMENT = {
    "anger": "negative",
    "sadness": "negative",
    "fear": "negative",
    "disgust": "negative",
    "happiness": "positive",
    "neutral": "neutral",
}

SENTIMENT_SCORES = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def fuse_segment(
    text_emotion: Dict[str, Any],
    acoustic_emotion: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Confidence-gated fusion of text-based and acoustic emotion predictions.

    Args:
        text_emotion: Dict with keys 'emotion' (str), 'confidence' (float 0-1),
                      'sentiment' (str, optional), 'logits' (array, optional).
        acoustic_emotion: Dict with keys 'emotion' (str), 'confidence' (float 0-1),
                          'indeterminate' (bool, optional).

    Returns:
        Dict with fused 'emotion', 'sentiment', 'confidence', 'source', 'text_weight', 'acoustic_weight'.
    """
    text_conf = float(text_emotion.get("confidence", 0.0))
    acoustic_conf = float(acoustic_emotion.get("confidence", 0.0))
    text_emo = str(text_emotion.get("emotion", "neutral")).lower()
    acoustic_emo = str(acoustic_emotion.get("emotion", "neutral")).lower()
    indeterminate = bool(acoustic_emotion.get("indeterminate", False))

    if indeterminate and acoustic_conf > TEXT_CONF_THRESHOLD:
        acoustic_conf *= 0.5

    if text_conf > TEXT_CONF_THRESHOLD and text_conf >= acoustic_conf:
        return {
            "emotion": text_emo,
            "sentiment": EMOTION_TO_SENTIMENT.get(text_emo, "neutral"),
            "confidence": text_conf,
            "source": "text",
            "text_weight": 1.0,
            "acoustic_weight": 0.0,
        }

    if acoustic_conf > ACOUSTIC_CONF_THRESHOLD and acoustic_conf > text_conf:
        return {
            "emotion": acoustic_emo,
            "sentiment": EMOTION_TO_SENTIMENT.get(acoustic_emo, "neutral"),
            "confidence": acoustic_conf,
            "source": "acoustic",
            "text_weight": 0.0,
            "acoustic_weight": 1.0,
        }

    text_score = SENTIMENT_SCORES.get(EMOTION_TO_SENTIMENT.get(text_emo, "neutral"), 0.0) * text_conf
    acoustic_score = SENTIMENT_SCORES.get(EMOTION_TO_SENTIMENT.get(acoustic_emo, "neutral"), 0.0) * acoustic_conf
    total_conf = text_conf + acoustic_conf
    if total_conf == 0:
        total_conf = 1e-8

    w_text = TEXT_WEIGHT * (text_conf / total_conf) * 2
    w_acoustic = ACOUSTIC_WEIGHT * (acoustic_conf / total_conf) * 2
    w_sum = w_text + w_acoustic
    if w_sum > 0:
        w_text /= w_sum
        w_acoustic /= w_sum
    else:
        w_text, w_acoustic = TEXT_WEIGHT, ACOUSTIC_WEIGHT

    blended_score = w_text * text_score + w_acoustic * acoustic_score

    if blended_score > 0.3:
        fused_sentiment = "positive"
    elif blended_score < -0.3:
        fused_sentiment = "negative"
    else:
        fused_sentiment = "neutral"

    fused_confidence = w_text * text_conf + w_acoustic * acoustic_conf

    if w_text > w_acoustic:
        fused_emotion = text_emo
    elif w_acoustic > w_text:
        fused_emotion = acoustic_emo
    else:
        fused_emotion = text_emo

    return {
        "emotion": fused_emotion,
        "sentiment": fused_sentiment,
        "confidence": round(fused_confidence, 4),
        "source": "fused",
        "text_weight": round(w_text, 4),
        "acoustic_weight": round(w_acoustic, 4),
    }


def fuse_segments(
    text_results: List[Dict[str, Any]],
    acoustic_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Fuse parallel lists of text and acoustic emotion results per segment.

    Args:
        text_results: List of text emotion dicts, one per segment.
        acoustic_results: List of acoustic emotion dicts, one per segment.

    Returns:
        List of fused emotion dicts, one per segment.
    """
    n = min(len(text_results), len(acoustic_results))
    fused = []
    for i in range(n):
        fused.append(fuse_segment(text_results[i], acoustic_results[i]))
    for i in range(n, max(len(text_results), len(acoustic_results))):
        if i < len(text_results):
            emo = str(text_results[i].get("emotion", "neutral")).lower()
            fused.append({
                "emotion": emo,
                "sentiment": EMOTION_TO_SENTIMENT.get(emo, "neutral"),
                "confidence": float(text_results[i].get("confidence", 0.0)),
                "source": "text",
                "text_weight": 1.0,
                "acoustic_weight": 0.0,
            })
        elif i < len(acoustic_results):
            emo = str(acoustic_results[i].get("emotion", "neutral")).lower()
            fused.append({
                "emotion": emo,
                "sentiment": EMOTION_TO_SENTIMENT.get(emo, "neutral"),
                "confidence": float(acoustic_results[i].get("confidence", 0.0)),
                "source": "acoustic",
                "text_weight": 0.0,
                "acoustic_weight": 1.0,
            })
    return fused
