"""
Role inference orchestrator.

Loads the MuRIL-based role classifier, builds speaker-tagged transcript context
from diarized turns, predicts the role mapping (agent/customer), and applies it
to segments.

This module replaces the old static "first speaker = agent" convention with a
learned classifier that understands Hindi/Hinglish call semantics.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from transformers import AutoTokenizer

from svar.models.role_classifier import (
    SpeakerRoleClassifier,
    SpeakerRoleResult,
    apply_role_mapping,
    build_role_context,
    infer_roles,
)


@dataclass
class RoleResolution:
    """Result of role inference applied to a call."""
    role_mapping: Dict[str, str]  # {"spk_0": "agent", "spk_1": "customer"}
    result: Optional[SpeakerRoleResult] = None
    method: str = "classifier"  # "classifier" | "fallback" | "disabled"
    applied: bool = False


class RoleInferenceEngine:
    """Loads and runs the MuRIL role classifier with lazy model loading."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "google/muril-base-cased",
        device: str = "cuda",
    ):
        self._model_path = model_path
        self._model_name = model_name
        self._device = device
        self._model: Optional[SpeakerRoleClassifier] = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._loaded = False
        self._disabled = os.environ.get("ROLE_INFERENCE_DISABLED", "").lower() in ("1", "true", "yes")

    @property
    def enabled(self) -> bool:
        return not self._disabled

    def _ensure_loaded(self):
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                self._model = SpeakerRoleClassifier(model_name=self._model_name)

                if self._model_path and Path(self._model_path).exists():
                    state = torch.load(self._model_path, map_location="cpu", weights_only=True)
                    if "model_state_dict" in state:
                        self._model.load_state_dict(state["model_state_dict"])
                    else:
                        self._model.load_state_dict(state)
                    print(f"[role_inference] Loaded trained weights from {self._model_path}")
                else:
                    print(f"[role_inference] Using untrained MuRIL classifier (no checkpoint found at {self._model_path})")

                self._model = self._model.to(self._device).eval()
                self._loaded = True
            except Exception as e:
                print(f"[role_inference] Failed to load model: {e}")
                self._model = None
                self._tokenizer = None

    def resolve(self, segments: list) -> RoleResolution:
        """Run role inference on a list of segments and return the resolution.

        Args:
            segments: List of segment dicts with 'speaker', 'text', 'start' keys.

        Returns:
            RoleResolution with role_mapping (spk_0/spk_1 → agent/customer).
        """
        if not self.enabled:
            return RoleResolution(role_mapping={}, method="disabled")

        spk_speakers = sorted(set(
            seg.get("speaker", "") for seg in segments
            if seg.get("speaker", "").startswith("spk_")
        ))

        if len(spk_speakers) < 2:
            if len(spk_speakers) == 1:
                return RoleResolution(
                    role_mapping={spk_speakers[0]: "agent"},
                    method="fallback",
                )
            return RoleResolution(role_mapping={}, method="fallback")

        self._ensure_loaded()

        if self._model is None or self._tokenizer is None:
            heuristic = self._heuristic_resolve(segments, spk_speakers)
            return RoleResolution(
                role_mapping=heuristic,
                method="heuristic",
                applied=bool(heuristic),
            )

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            result = infer_roles(
                model=self._model,
                tokenizer=self._tokenizer,
                diarized_turns=segments,
                device=device,
            )
            mapping = apply_role_mapping(segments, result)

            if not any(v != "unknown" for v in mapping.values()):
                heuristic = self._heuristic_resolve(segments, spk_speakers)
                if heuristic:
                    return RoleResolution(
                        role_mapping=heuristic,
                        method="heuristic",
                        applied=True,
                    )

            return RoleResolution(
                role_mapping=mapping,
                result=result,
                method="classifier",
                applied=True,
            )
        except Exception as e:
            print(f"[role_inference] Classifier failed: {e}")
            heuristic = self._heuristic_resolve(segments, spk_speakers)
            return RoleResolution(
                role_mapping=heuristic,
                method="heuristic",
                applied=bool(heuristic),
            )

    def _heuristic_resolve(self, segments: list, spk_speakers: list) -> dict:
        """Heuristic role resolution based on self-introduction patterns.

        Looks for the speaker who introduces themselves as calling from a company
        or mentions job/recruitment keywords — that speaker is the agent.
        """
        import re

        agent_patterns = re.compile(
            r"(?:बात\s+कर\s+रह[ीा]\s+हूं|"
            r"जॉब\s+रिकॉर्डिंग|"
            r"कैलिबर|"
            r"इंटरव्यू|"
            r"रिक्वायरमेंट|"
            r"सेल[्स]?\s+प्रोफाइल|"
            r"सेलेक्ट\s+होते|"
            r"सैलरी|"
            r"सैलरी\s+होगा|"
            r"आपको\s+(?:बता|समझा|एक्सप्लेन)|"
            r"सुनो\s+आप\s+पहले|"
            r"ओके\s+अ|"
            r"गुड\s+आफ्टरनून|"
            r"नंबर\s+दिया\s+है)",
            re.IGNORECASE,
        )

        spk_text = {}
        for seg in segments:
            spk = seg.get("speaker", "")
            text = seg.get("text", "")
            if spk.startswith("spk_") and text:
                spk_text.setdefault(spk, []).append(text)

        for spk in spk_speakers:
            full_text = " ".join(spk_text.get(spk, []))
            if agent_patterns.search(full_text):
                other = [s for s in spk_speakers if s != spk]
                if other:
                    return {spk: "agent", other[0]: "customer"}

        return {}

    def apply_mapping(self, segments: list, resolution: RoleResolution) -> list:
        """Apply role mapping to segments, replacing spk_0/spk_1 with agent/customer.

        Args:
            segments: List of segment dicts (mutated in place).
            resolution: RoleResolution from resolve().

        Returns:
            The same list, with speaker fields updated.
        """
        mapping = resolution.role_mapping
        if not mapping:
            return segments

        for seg in segments:
            spk = seg.get("speaker", "")
            if spk in mapping and mapping[spk] != "unknown":
                seg["speaker"] = mapping[spk]

        return segments


_engine: Optional[RoleInferenceEngine] = None
_engine_lock = threading.Lock()


def get_role_engine(
    model_path: Optional[str] = None,
    model_name: str = "google/muril-base-cased",
    device: str = "cuda",
) -> RoleInferenceEngine:
    """Get or create the singleton RoleInferenceEngine."""
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        _engine = RoleInferenceEngine(
            model_path=model_path,
            model_name=model_name,
            device=device,
        )
        return _engine
