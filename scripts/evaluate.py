"""
Evaluation script with call-disjoint metrics.

All metrics are computed at the call level (not turn level) to prevent
data leakage from the same call appearing in both train and test.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

import torch
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)


from svar.schemas import EMOINHINDI_EMOTIONS


def compute_emotion_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute multi-label emotion metrics."""
    y_pred_binary = (y_pred > threshold).astype(int)

    metrics = {
        "micro_f1": f1_score(y_true, y_pred_binary, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred_binary, average="macro", zero_division=0),
        "micro_precision": precision_score(y_true, y_pred_binary, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred_binary, average="micro", zero_division=0),
    }

    # Per-emotion F1
    per_emotion = {}
    for i, emo in enumerate(EMOINHINDI_EMOTIONS):
        if i < y_true.shape[1]:
            per_emotion[f"f1_{emo}"] = f1_score(
                y_true[:, i], y_pred_binary[:, i], zero_division=0,
            )
    metrics["per_emotion_f1"] = per_emotion

    return metrics


def compute_sentiment_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute multi-class sentiment metrics."""
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def compute_call_level_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute all metrics at the call level.

    Args:
        predictions: List of turn predictions with model outputs.
        ground_truth: List of turn ground truth with annotations.

    Returns:
        Dict with all metrics.
    """
    # Group by call_id
    pred_by_call = defaultdict(list)
    gt_by_call = defaultdict(list)

    for pred in predictions:
        call_id = pred.get("call_id", "unknown")
        pred_by_call[call_id].append(pred)

    for gt in ground_truth:
        call_id = gt.get("call_id", "unknown")
        gt_by_call[call_id].append(gt)

    # Compute metrics per call, then average
    all_emotion_true = []
    all_emotion_pred = []
    all_sentiment_true = []
    all_sentiment_pred = []

    for call_id in pred_by_call:
        if call_id not in gt_by_call:
            continue

        call_preds = sorted(pred_by_call[call_id], key=lambda x: x.get("start", 0))
        call_gts = sorted(gt_by_call[call_id], key=lambda x: x.get("start", 0))

        # Align predictions with ground truth
        for pred, gt in zip(call_preds, call_gts):
            # Emotion
            pred_emotions = pred.get("emotion", {})
            gt_emotions = gt.get("emotion_label", [])

            pred_vec = np.zeros(len(EMOINHINDI_EMOTIONS))
            for emo, prob in pred_emotions.items():
                if emo in EMOINHINDI_EMOTIONS:
                    pred_vec[EMOINHINDI_EMOTIONS.index(emo)] = prob

            gt_vec = np.zeros(len(EMOINHINDI_EMOTIONS))
            for emo in gt_emotions:
                if emo in EMOINHINDI_EMOTIONS:
                    gt_vec[EMOINHINDI_EMOTIONS.index(emo)] = 1.0

            all_emotion_true.append(gt_vec)
            all_emotion_pred.append(pred_vec)

            # Sentiment
            pred_sentiment = pred.get("sentiment", "neutral")
            gt_sentiment = gt.get("sentiment", "neutral")

            sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
            all_sentiment_true.append(sentiment_map.get(gt_sentiment, 1))
            all_sentiment_pred.append(sentiment_map.get(pred_sentiment, 1))

    # Compute metrics
    results = {}

    if all_emotion_true:
        emotion_true = np.array(all_emotion_true)
        emotion_pred = np.array(all_emotion_pred)
        results["emotion"] = compute_emotion_metrics(emotion_true, emotion_pred)

    if all_sentiment_true:
        sentiment_true = np.array(all_sentiment_true)
        sentiment_pred = np.array(all_sentiment_pred)
        results["sentiment"] = compute_sentiment_metrics(sentiment_true, sentiment_pred)

    # Call-level counts
    results["n_calls"] = len(set(pred_by_call.keys()) & set(gt_by_call.keys()))
    results["n_turns"] = len(all_emotion_true)

    return results


def print_report(results: Dict[str, Any]) -> None:
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("SVAR Emotion Model Evaluation Report")
    print("=" * 60)

    print(f"\nCalls evaluated: {results.get('n_calls', 0)}")
    print(f"Turns evaluated: {results.get('n_turns', 0)}")

    if "emotion" in results:
        emo = results["emotion"]
        print(f"\n--- Emotion Metrics ---")
        print(f"  Micro F1:        {emo['micro_f1']:.4f}")
        print(f"  Macro F1:        {emo['macro_f1']:.4f}")
        print(f"  Micro Precision: {emo['micro_precision']:.4f}")
        print(f"  Micro Recall:    {emo['micro_recall']:.4f}")

        if "per_emotion_f1" in emo:
            print(f"\n  Per-emotion F1:")
            for emo_name, f1 in emo["per_emotion_f1"].items():
                print(f"    {emo_name:20s}: {f1:.4f}")

    if "sentiment" in results:
        sent = results["sentiment"]
        print(f"\n--- Sentiment Metrics ---")
        print(f"  Accuracy:     {sent['accuracy']:.4f}")
        print(f"  Macro F1:     {sent['macro_f1']:.4f}")
        print(f"  Weighted F1:  {sent['weighted_f1']:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)
    with open(args.ground_truth) as f:
        ground_truth = json.load(f)

    results = compute_call_level_metrics(predictions, ground_truth)
    print_report(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
