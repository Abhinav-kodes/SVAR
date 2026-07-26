"""
Gemini Flash-based role resolver.

Sends the diarized transcript to Gemini and asks it to identify
which speaker is the agent and which is the customer, based on
conversation context, language patterns, and call structure.
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

_SYSTEM_PROMPT = """You are an expert at analyzing Hindi/Hinglish call center transcripts.

Your task: identify which speaker is the AGENT (call center representative) and which is the CUSTOMER.

CLUES TO LOOK FOR:
- The agent typically introduces themselves, mentions company name, offers services/products
- The agent uses formal/professional language: "सर/मैडम", "हमारी कंपनी", "आपका अकाउंट"
- The agent asks for personal details, explains processes, follows scripts
- The customer typically asks questions, expresses concerns, provides personal info
- The customer may sound confused, frustrated, or inquisitive
- The agent says things like "बात कर रहा हूं", "गुड आफ्टरनून", "वेलकम"
- In Hindi calls, the agent often starts with a greeting or self-introduction
- The first speaker is usually the agent (but not always)

EDGE CASES:
- If both speakers sound like agents (e.g., two agents talking), pick the one who initiated the call topic
- If it's a casual conversation (not a call center call), make your best guess based on who seems to be leading
- If you truly cannot determine, say "uncertain"

OUTPUT: Return ONLY a valid JSON object. No markdown fences, no explanation outside JSON."""


def _load_keys() -> List[str]:
    if _KEYS_PATH.exists():
        data = json.loads(_KEYS_PATH.read_text())
        return data.get("keys", [])
    return []


class GeminiRoleResolver:
    """Thread-safe Gemini Flash role resolver with round-robin key rotation."""

    def __init__(self):
        self._keys = _load_keys()
        self._idx = 0
        self._lock = threading.Lock()
        self._available = len(self._keys) > 0
        self._disabled = os.environ.get("ROLE_INFERENCE_DISABLED", "").lower() in ("1", "true", "yes")

    @property
    def enabled(self) -> bool:
        return self._available and not self._disabled

    def _next_key(self) -> str:
        with self._lock:
            key = self._keys[self._idx % len(self._keys)]
            self._idx += 1
        return key

    def resolve(self, segments: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        """Identify agent/customer roles from a transcript using Gemini.

        Args:
            segments: List of segment dicts with 'speaker', 'text' keys.

        Returns:
            Dict mapping speaker IDs to roles, e.g. {"spk_0": "agent", "spk_1": "customer"},
            or None if Gemini fails or is disabled.
        """
        if not self.enabled:
            return None

        speakers = sorted(set(s.get("speaker", "") for s in segments if s.get("speaker")))
        if len(speakers) < 2:
            return None

        transcript_lines = []
        for seg in segments:
            spk = seg.get("speaker", "unknown")
            text = seg.get("text", "").strip()
            if text:
                transcript_lines.append(f"[{spk}]: {text}")

        if not transcript_lines:
            return None

        transcript = "\n".join(transcript_lines[:80])

        prompt = f"""Here is a call transcript with speaker labels:

{transcript}

Analyze the conversation and identify which speaker is the AGENT and which is the CUSTOMER.

Return a JSON object like:
{{"spk_0": "agent", "spk_1": "customer"}}

Or if uncertain:
{{"spk_0": "uncertain", "spk_1": "uncertain"}}

Only include the speaker IDs found in the transcript."""

        raw = self._call_gemini(prompt)
        return self._parse_result(raw, speakers)

    def _call_gemini(self, prompt: str) -> str:
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
                            max_output_tokens=1024,
                        ),
                    )
                    if response.text:
                        return response.text
                except Exception as e:
                    last_err = e
                    if "429" in str(e):
                        continue
                    break

        print(f"[role_llm] Gemini failed across all models/keys: {last_err}")
        return "{}"

    def _parse_result(self, raw: str, expected_speakers: List[str]) -> Optional[Dict[str, str]]:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            print(f"[role_llm] Failed to parse LLM output: {text[:200]}")
            return None

        if not isinstance(parsed, dict):
            return None

        mapping = {}
        for spk in expected_speakers:
            role = parsed.get(spk, "unknown")
            if role in ("agent", "customer"):
                mapping[spk] = role

        if not mapping or all(v == "unknown" for v in mapping.values()):
            return None

        if len(mapping) == 1:
            assigned = list(mapping.values())[0]
            remaining = [s for s in expected_speakers if s not in mapping]
            if remaining:
                mapping[remaining[0]] = "customer" if assigned == "agent" else "agent"

        return mapping
