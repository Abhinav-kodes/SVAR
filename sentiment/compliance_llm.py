"""
Gemini Flash-based compliance checker.

Replaces regex/keyword false-positive-prone matching with LLM context understanding.
Uses key rotation across multiple API keys for rate-limit resilience.
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

_system_prompt = """You are a compliance auditor for Indian call centers regulated by RBI and IRDAI.

Your job: analyze each text segment and determine if it contains compliance violations.

RULES:
1. ABUSIVE LANGUAGE: Flag actual insults, profanity, derogatory terms (Hindi or English).
   - "कमीना", "साला", "हरामी", "भोसड़ी", "लौड़ा", "कुत्ता" (used as insult) = abusive
   - "कमीज" (shirt), "कितने" (how many), "बेचना" (to sell), "भैया" (brother), "पड़ोसे" (neighbor) = NOT abusive
   - Context matters: "तू कमीना है" = abusive, "मैं कमीज पहनता हूं" = not abusive
2. RBI VIOLATIONS: Sharing PIN, OTP, bank account details, CVV, Aadhaar number inappropriately.
   - "बैंक खाता नंबर बताओ" = violation
   - "बैंक में काम करता हूं" = NOT a violation (just mentioning bank)
3. IRDAI VIOLATIONS: Insurance misrepresentation, fraud, unsolicited calls, mis-selling.

OUTPUT: Return ONLY a valid JSON array, one object per input segment. No markdown, no explanation outside JSON.
"""


def _load_keys() -> List[str]:
    if _KEYS_PATH.exists():
        data = json.loads(_KEYS_PATH.read_text())
        return data.get("keys", [])
    return []


class GeminiComplianceChecker:
    """Thread-safe Gemini Flash compliance checker with round-robin key rotation."""

    def __init__(self):
        self._keys = _load_keys()
        self._idx = 0
        self._lock = threading.Lock()
        self._available = len(self._keys) > 0
        self._disabled = os.environ.get("LLM_COMPLIANCE_DISABLED", "").lower() in ("1", "true", "yes")

    @property
    def enabled(self) -> bool:
        return self._available and not self._disabled

    def _next_key(self) -> str:
        with self._lock:
            key = self._keys[self._idx % len(self._keys)]
            self._idx += 1
        return key

    def check_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check a batch of segments for compliance violations using Gemini Flash.

        Args:
            segments: List of dicts with 'text', 'speaker', 'start' keys.

        Returns:
            List of dicts, one per segment, each with:
                'abusive_matches': [{'keyword', 'type': 'llm', 'context_note'}]
                'rbi_violations': [{'pattern': 'llm', 'match': keyword, 'context_note'}]
                'irdai_violations': [{'pattern': 'llm', 'match': keyword, 'context_note'}]
                'compliant': bool
                'flags': list[str]
        """
        if not self.enabled:
            return []

        batch_input = []
        for i, seg in enumerate(segments):
            text = seg.get("text", "")
            if not text.strip():
                batch_input.append({"index": i, "text": "", "speaker": seg.get("speaker", "unknown")})
                continue
            batch_input.append({
                "index": i,
                "text": text,
                "speaker": seg.get("speaker", "unknown"),
            })

        results_raw = self._call_gemini(batch_input)
        return self._parse_results(results_raw, segments)

    def _call_gemini(self, batch_input: List[Dict]) -> str:
        user_msg = json.dumps(batch_input, ensure_ascii=False)

        prompt = f"""Analyze these call center transcript segments for compliance violations.

{_system_prompt}

Input segments (JSON array):
{user_msg}

Return a JSON array with one object per input segment (same order, same index field).
Each object: {{"index": <int>, "violations": [{{"type": "abusive"|"rbi"|"irdai", "keyword": "<word/phrase>", "context_note": "<brief reason>"}}], "compliant": <bool>}}"""

        last_err = None
        for model in _FALLBACK_MODELS:
            for attempt in range(len(self._keys)):
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
                        return response.text
                except Exception as e:
                    last_err = e
                    if "429" in str(e):
                        continue
                    break

        print(f"[compliance_llm] Gemini failed across all models/keys: {last_err}")
        return "[]"

    def _parse_results(self, raw: str, segments: List[Dict]) -> List[Dict[str, Any]]:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Attempt to fix truncated JSON array by finding last closed object
            last_brace = text.rfind("}")
            if last_brace != -1:
                truncated = text[:last_brace+1] + "]"
                try:
                    parsed = json.loads(truncated)
                except Exception:
                    print(f"[compliance_llm] Failed to parse LLM output: {text[:200]}")
                    return []
            else:
                print(f"[compliance_llm] Failed to parse LLM output: {text[:200]}")
                return []

        if not isinstance(parsed, list):
            return []


        by_index = {}
        for item in parsed:
            idx = item.get("index")
            if idx is not None and 0 <= idx < len(segments):
                by_index[idx] = item

        results = []
        for i, seg in enumerate(segments):
            item = by_index.get(i, {})
            violations = item.get("violations", [])

            abusive = []
            rbi = []
            irdai = []
            flags = []

            for v in violations:
                vtype = v.get("type", "")
                kw = v.get("keyword", "")
                note = v.get("context_note", "")

                if vtype == "abusive":
                    abusive.append({"keyword": kw, "type": "llm", "context_note": note})
                    flags.append(f"ABUSIVE:{kw}")
                elif vtype == "rbi":
                    rbi.append({"pattern": "llm", "match": kw, "context_note": note})
                    flags.append(f"RBI:{kw}")
                elif vtype == "irdai":
                    irdai.append({"pattern": "llm", "match": kw, "context_note": note})
                    flags.append(f"IRDAI:{kw}")

            compliant = item.get("compliant", len(flags) == 0)

            results.append({
                "abusive_matches": abusive,
                "regulatory": {
                    "rbi_violations": rbi,
                    "irdai_violations": irdai,
                    "total_violations": len(rbi) + len(irdai),
                },
                "compliant": compliant,
                "flags": flags,
                "violation_count": len(flags),
                "speaker": seg.get("speaker", "unknown"),
                "start": seg.get("start", seg.get("start_time_s", 0)),
                "end": seg.get("end", seg.get("end_time_s", 0)),
            })

        return results


checker = GeminiComplianceChecker()


def check_compliance_llm(segments: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Check segments using Gemini Flash. Returns None if unavailable."""
    if not checker.enabled:
        return None
    return checker.check_segments(segments)
