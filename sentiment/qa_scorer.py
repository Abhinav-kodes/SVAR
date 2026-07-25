import os
import yaml
from typing import Dict, Any, List, Optional

DEFAULT_WEIGHTS = {
    "customer_sentiment": 0.30,
    "compliance": 0.25,
    "agent_stability": 0.20,
    "intent_resolution": 0.15,
    "talk_ratio": 0.10,
}

GRADE_THRESHOLDS = [
    (85, "A"),
    (70, "B"),
    (55, "C"),
    (0, "D"),
]

SENTIMENT_SCORES = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}


def load_weights(weights_path: Optional[str] = None) -> Dict[str, float]:
    if weights_path and os.path.exists(weights_path):
        with open(weights_path, "r") as f:
            cfg = yaml.safe_load(f)
        if isinstance(cfg, dict) and "weights" in cfg:
            return cfg["weights"]
        return cfg
    return DEFAULT_WEIGHTS.copy()


def compute_customer_sentiment_score(emotions: List[Dict[str, Any]]) -> float:
    """Average customer sentiment mapped to 0-100."""
    customer_emotions = [e for e in emotions if "customer" in str(e.get("speaker", "")).lower()]
    if not customer_emotions:
        return 50.0
    scores = []
    for e in customer_emotions:
        sent = str(e.get("sentiment", "neutral")).lower()
        conf = float(e.get("confidence", 1.0))
        base = SENTIMENT_SCORES.get(sent, 0.5)
        scores.append(base * conf + 0.5 * (1 - conf))
    return round((sum(scores) / len(scores)) * 100, 1)


def compute_compliance_score(compliance_result: Dict[str, Any]) -> float:
    """100 if fully compliant, penalized by violation count."""
    total = compliance_result.get("total_violations", 0)
    agent_v = compliance_result.get("agent_violations", 0)
    if total == 0:
        return 100.0
    penalty = min(total * 10 + agent_v * 5, 100)
    return round(max(0, 100 - penalty), 1)


def compute_agent_stability_score(emotions: List[Dict[str, Any]]) -> float:
    """Measure agent emotional consistency — low variance = high score."""
    agent_emotions = [e for e in emotions if "agent" in str(e.get("speaker", "")).lower()]
    if not agent_emotions:
        return 75.0
    sentiments = [SENTIMENT_SCORES.get(str(e.get("sentiment", "neutral")).lower(), 0.5) for e in agent_emotions]
    if len(sentiments) < 2:
        return 80.0
    mean_s = sum(sentiments) / len(sentiments)
    variance = sum((s - mean_s) ** 2 for s in sentiments) / len(sentiments)
    stability = max(0, 1 - variance * 4)
    return round(stability * 100, 1)


def compute_intent_resolution_score(compliance_result: Dict[str, Any], emotions: List[Dict[str, Any]]) -> float:
    """Heuristic: no violations + positive resolution signals = high score."""
    score = 70.0
    if compliance_result.get("compliant", True):
        score += 15.0
    all_emotions = [str(e.get("sentiment", "neutral")).lower() for e in emotions]
    pos_ratio = sum(1 for s in all_emotions if s == "positive") / max(len(all_emotions), 1)
    score += pos_ratio * 15
    return round(min(score, 100), 1)


def compute_talk_ratio(segments: List[Dict[str, Any]]) -> float:
    """Compute agent:customer talk time ratio. Ideal ~0.4-0.6."""
    agent_time = 0.0
    customer_time = 0.0
    for seg in segments:
        dur = float(seg.get("end", seg.get("end_time_s", 0))) - float(seg.get("start", seg.get("start_time_s", 0)))
        speaker = str(seg.get("speaker", "")).lower()
        if "agent" in speaker:
            agent_time += dur
        else:
            customer_time += dur
    total = agent_time + customer_time
    if total == 0:
        return 50.0
    agent_pct = (agent_time / total) * 100
    if 35 <= agent_pct <= 65:
        return 100.0
    deviation = min(abs(agent_pct - 50), 50)
    return round(max(0, 100 - deviation * 2), 1)


def compute_talk_ratio_score(segments: List[Dict[str, Any]]) -> float:
    """Score based on how close to ideal talk ratio."""
    return compute_talk_ratio(segments)


def score_call(
    segments: List[Dict[str, Any]],
    emotions: List[Dict[str, Any]],
    compliance_result: Dict[str, Any],
    weights_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute QA score for a full call.

    Args:
        segments: List of diarized segments with speaker labels.
        emotions: List of fused emotion results per segment.
        compliance_result: Output from compliance_engine.analyze_call().
        weights_path: Optional YAML path for custom weights.

    Returns:
        Dict with component scores, total score, grade, and breakdown.
    """
    weights = load_weights(weights_path)

    cust_sent = compute_customer_sentiment_score(emotions)
    comp_score = compute_compliance_score(compliance_result)
    agent_stab = compute_agent_stability_score(emotions)
    intent_res = compute_intent_resolution_score(compliance_result, emotions)
    talk_ratio = compute_talk_ratio_score(segments)

    components = {
        "customer_sentiment": cust_sent,
        "compliance": comp_score,
        "agent_stability": agent_stab,
        "intent_resolution": intent_res,
        "talk_ratio": talk_ratio,
    }

    total = sum(
        weights.get(k, 0) * v for k, v in components.items()
    )
    total = round(min(total, 100), 1)

    grade = "D"
    for threshold, g in GRADE_THRESHOLDS:
        if total >= threshold:
            grade = g
            break

    return {
        "qa_score": total,
        "grade": grade,
        "components": components,
        "weights_used": weights,
        "total_violations": compliance_result.get("total_violations", 0),
    }
