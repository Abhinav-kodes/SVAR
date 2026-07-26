"""
Unified Gemini Flash Unified Audit Engine.

Combines Compliance Audit, QA Scoring, and CRM Note Generation into ONE single Gemini API call.
Saves 66% API quota & latency while guaranteeing complete analytical consistency across all metrics.
"""

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class GeminiAuditEngine:
    """Thread-safe Gemini Flash engine performing Compliance, QA Scoring & CRM Generation in ONE call."""

    def __init__(self):
        self._keys = _load_keys()
        self._idx = 0
        self._lock = threading.Lock()
        self._available = len(self._keys) > 0
        self._disabled = os.environ.get("LLM_AUDIT_DISABLED", "").lower() in ("1", "true", "yes")

    @property
    def enabled(self) -> bool:
        return self._available and not self._disabled

    def _next_key(self) -> str:
        with self._lock:
            key = self._keys[self._idx % len(self._keys)]
            self._idx += 1
        return key

    def audit_call(
        self,
        segments: List[Dict[str, Any]],
        talk_ratio_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Perform unified audit: Compliance + QA Scorecard + CRM Note in 1 Gemini call."""
        if not self.enabled or not segments:
            return None

        batch_input = []
        for i, seg in enumerate(segments):
            batch_input.append({
                "index": i,
                "speaker": seg.get("speaker", "unknown"),
                "text": seg.get("text", "").strip(),
                "start": seg.get("start_time_s", seg.get("start", 0)),
                "end": seg.get("end_time_s", seg.get("end", 0)),
            })

        user_msg = json.dumps(batch_input, ensure_ascii=False)

        prompt = f"""You are a Lead AI Quality & Compliance Auditor for an Indian enterprise call center (regulated by RBI & IRDAI).
Analyze the following diarized Hindi/English call transcript and generate a UNIFIED AUDIT REPORT covering:
1. Compliance Violations (RBI, IRDAI, Abusive Language, Payment/Personal Threats).
2. Automated QA Scorecard (0-100 Overall Score, Grade A/B/C/D/F, and Category Ratings out of 100).
3. Executive CRM Audit Note (Executive Summary, Key Discussion Points, Compliance Overview, Recommended Next Action).

TRANSCRIPT SEGMENTS (JSON array):
{user_msg}

INSTRUCTIONS:
Return ONLY a valid JSON object with EXACTLY three top-level keys: "compliance", "qa", and "crm_note".
Strict format structure required:

{{
  "compliance": {{
    "compliant": boolean,
    "total_violations": int,
    "agent_violations": int,
    "customer_violations": int,
    "segment_results": [
      {{
        "index": int,
        "speaker": string,
        "violation_count": int,
        "flags": ["RBI:<details>" | "IRDAI:<details>" | "ABUSIVE:<details>" | "THREAT:<details>"]
      }}
    ]
  }},
  "qa": {{
    "qa_score": float (0-100),
    "grade": "A" | "B" | "C" | "D" | "F",
    "components": {{
      "customer_sentiment": float (0-100),
      "compliance": float (0-100),
      "agent_stability": float (0-100),
      "intent_resolution": float (0-100),
      "talk_ratio": float (0-100)
    }},
    "weights_used": {{
      "customer_sentiment": 0.30,
      "compliance": 0.25,
      "agent_stability": 0.20,
      "intent_resolution": 0.15,
      "talk_ratio": 0.10
    }}
  }},
  "crm_note": {{
    "summary": string (2-3 concise sentences in English),
    "key_points": [string] (3-5 bullet point strings),
    "compliance_summary": string (1-2 sentences on regulatory adherence),
    "recommended_action": string (1 actionable sentence)
  }}
}}

Output ONLY raw valid JSON. No markdown backticks, no text outside the JSON.
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
                            temperature=0.1,
                            max_output_tokens=8192,
                        ),
                    )
                    if response.text:
                        raw = response.text.strip()
                        if raw.startswith("```"):
                            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

                        parsed = json.loads(raw)
                        if isinstance(parsed, dict) and "compliance" in parsed and "qa" in parsed and "crm_note" in parsed:
                            parsed["unified_llm"] = True
                            return parsed
                except Exception as e:
                    last_err = e
                    if "429" in str(e):
                        continue
                    break

        print(f"[audit_engine] Unified Gemini Audit failed across models: {last_err}")
        return None


_audit_engine = GeminiAuditEngine()


def run_unified_audit(
    segments: List[Dict[str, Any]],
    talk_ratio_data: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Execute unified Gemini audit returning compliance, qa, and crm_note in one call."""
    return _audit_engine.audit_call(segments, talk_ratio_data)
