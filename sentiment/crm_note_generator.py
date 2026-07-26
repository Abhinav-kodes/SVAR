import os
import json
import math
import re
import threading
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Optional

from google import genai
from google.genai import types

_MODEL = "gemini-3.5-flash-lite"
_FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash"]
_KEYS_PATH = Path(__file__).resolve().parent.parent / "credentials" / "gemini-api-keys.json"


def _load_keys() -> List[str]:
    if _KEYS_PATH.exists():
        try:
            data = json.loads(_KEYS_PATH.read_text())
            return data.get("keys", [])
        except Exception:
            pass
    return []


class GeminiCRMNoteGenerator:
    """Thread-safe Gemini Flash CRM note generator with round-robin key rotation."""

    def __init__(self):
        self._keys = _load_keys()
        self._idx = 0
        self._lock = threading.Lock()
        self._available = len(self._keys) > 0
        self._disabled = os.environ.get("LLM_CRM_DISABLED", "").lower() in ("1", "true", "yes")

    @property
    def enabled(self) -> bool:
        return self._available and not self._disabled

    def _next_key(self) -> str:
        with self._lock:
            key = self._keys[self._idx % len(self._keys)]
            self._idx += 1
        return key

    def generate(
        self,
        transcript: str,
        emotions: Optional[List[Dict[str, Any]]] = None,
        compliance_result: Optional[Dict[str, Any]] = None,
        qa_result: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        flags = compliance_result.get("flags", []) if compliance_result else []
        total_v = compliance_result.get("total_violations", 0) if compliance_result else 0
        qa_score = qa_result.get("qa_score", "N/A") if qa_result else "N/A"
        grade = qa_result.get("grade", "N/A") if qa_result else "N/A"

        prompt = f"""You are an executive Call Quality & CRM Auditor for an Indian enterprise call center.
Analyze the following Hindi/English call transcript along with the compliance and QA findings, and produce a structured CRM Audit Note.

TRANSCRIPT:
{transcript}

COMPLIANCE FINDINGS:
Total Violations: {total_v}
Flags: {json.dumps(flags, ensure_ascii=False)}

QA EVALUATION:
Score: {qa_score}
Grade: {grade}

INSTRUCTIONS:
Generate a clean JSON object with EXACTLY these four fields:
1. "summary": A concise 2-3 sentence overview in English summarizing the call context, customer query/issue, and outcome.
2. "key_points": A list of 3-5 bullet point strings detailing key discussions, customer sentiment, agent tone, and main takeaways.
3. "compliance_summary": A 1-2 sentence assessment of regulatory compliance (RBI/IRDAI) and conduct.
4. "recommended_action": A clear, single-sentence actionable next step (e.g. agent coaching, supervisor escalation, customer callback, or no action).

Output ONLY valid JSON, no markdown code blocks, no extra commentary.
"""

        last_err = None
        for model in _FALLBACK_MODELS:
            for _ in range(len(self._keys)):
                key = self._next_key()
                try:
                    client = genai.Client(api_key=key)
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.2,
                            max_output_tokens=1024,
                        ),
                    )
                    if response.text:
                        raw = response.text.strip()
                        if raw.startswith("```"):
                            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict) and "summary" in parsed:
                            parsed["total_violations"] = total_v
                            parsed["llm_generated"] = True
                            return parsed
                except Exception as e:
                    last_err = e
                    if "429" in str(e):
                        continue
                    break

        print(f"[crm_llm] Gemini failed across models/keys: {last_err}")
        return None


_crm_generator = GeminiCRMNoteGenerator()


def _tokenize(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())


def tfidf_extractive_summary(
    transcript: str,
    emotions: List[Dict[str, Any]],
    compliance_flags: List[str],
    max_sentences: int = 5,
) -> str:
    """TF-IDF based extractive summary fallback."""
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
    use_api: bool = True,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a CRM note from call analysis, attempting Gemini LLM generation first.
    """
    if use_api:
        llm_note = _crm_generator.generate(transcript, emotions, compliance_result, qa_result)
        if llm_note:
            return llm_note

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
        "llm_generated": False,
    }
