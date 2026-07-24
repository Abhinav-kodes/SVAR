import os
import math
import re
from collections import Counter
from typing import Dict, Any, List, Optional


def _tokenize(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())


def tfidf_extractive_summary(
    transcript: str,
    emotions: List[Dict[str, Any]],
    compliance_flags: List[str],
    max_sentences: int = 5,
) -> str:
    """TF-IDF based extractive summarizer (no external API needed)."""
    sentences = re.split(r'(?<=[.!?।])\s+', transcript.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        sentences = [s.strip() for s in transcript.strip().split('\n') if s.strip()]

    if not sentences:
        return "No transcript available for summarization."

    tokenized = [_tokenize(s) for s in sentences]
    doc_freq = Counter()
    for tokens in tokenized:
        for t in set(tokens):
            doc_freq[t] += 1
    n_docs = len(sentences)

    scored = []
    for i, tokens in enumerate(tokenized):
        if not tokens:
            scored.append((0, i))
            continue
        tf = Counter(tokens)
        score = sum(
            (tf[t] / len(tokens)) * math.log((n_docs + 1) / (doc_freq[t] + 1)) + 1
            for t in set(tokens)
        )
        scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_indices = sorted([idx for _, idx in scored[:max_sentences]])
    summary = " ".join(sentences[i] for i in top_indices if i < len(sentences))

    parts = [summary]

    if compliance_flags:
        parts.append(f"\nCompliance flags: {'; '.join(compliance_flags[:5])}")

    if emotions:
        sentiment_counts = Counter(str(e.get("sentiment", "neutral")).lower() for e in emotions)
        dominant = sentiment_counts.most_common(1)[0][0] if sentiment_counts else "neutral"
        parts.append(f"Overall customer sentiment: {dominant}")

    return " ".join(parts)


def generate_crm_note(
    transcript: str,
    emotions: List[Dict[str, Any]],
    compliance_result: Dict[str, Any],
    qa_result: Optional[Dict[str, Any]] = None,
    use_api: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a CRM note from call analysis.

    Args:
        transcript: Full call transcript.
        emotions: List of fused emotion dicts per segment.
        compliance_result: Output from compliance_engine.analyze_call().
        qa_result: Optional QA scorer output.
        use_api: Whether to attempt Gemini API call.
        api_key: Gemini API key (or read from env GEMINI_API_KEY).

    Returns:
        Dict with 'summary', 'key_points', 'compliance_summary', 'recommended_action'.
    """
    flags = compliance_result.get("flags", []) if compliance_result else []
    total_v = compliance_result.get("total_violations", 0) if compliance_result else 0
    agent_v = compliance_result.get("agent_violations", 0) if compliance_result else 0

    summary = tfidf_extractive_summary(transcript, emotions, flags)

    key_points = []
    if emotions:
        sentiment_counts = Counter(str(e.get("sentiment", "neutral")).lower() for e in emotions)
        dominant = sentiment_counts.most_common(1)[0][0] if sentiment_counts else "neutral"
        key_points.append(f"Customer sentiment: {dominant}")

    agent_emotions = [e for e in emotions if "agent" in str(e.get("speaker", "")).lower()]
    if agent_emotions:
        agent_sentiments = Counter(str(e.get("sentiment", "neutral")).lower() for e in agent_emotions)
        agent_dom = agent_sentiments.most_common(1)[0][0]
        key_points.append(f"Agent tone: {agent_dom}")

    if qa_result:
        key_points.append(f"QA Score: {qa_result.get('qa_score', 'N/A')} ({qa_result.get('grade', 'N/A')})")

    compliance_summary = "No violations detected." if total_v == 0 else f"{total_v} violation(s) found ({agent_v} by agent)."

    if total_v == 0 and qa_result and qa_result.get("grade", "D") in ["A", "B"]:
        action = "No action required. Call met quality standards."
    elif total_v > 0 and agent_v > 0:
        action = "Flag for compliance review. Agent violations detected."
    elif qa_result and qa_result.get("grade") == "D":
        action = "Schedule coaching session. Low QA score."
    else:
        action = "Monitor. Call requires periodic review."

    return {
        "summary": summary,
        "key_points": key_points,
        "compliance_summary": compliance_summary,
        "recommended_action": action,
        "total_violations": total_v,
    }
