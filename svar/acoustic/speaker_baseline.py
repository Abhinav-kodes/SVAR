"""
Per-speaker baseline profile for within-call acoustic normalization.

The baseline captures each speaker's calm, early-call vocal characteristics.
Later turns are compared against this baseline to detect voice shifts.
The baseline is frozen after construction — it is NOT updated with
later (potentially angry) turns, which would normalize the anger away.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from ..schemas import SpeakerBaselineProfile


class SpeakerBaselineBuilder:
    """
    Builds a per-speaker baseline from early-call reference turns.

    Selection criteria for reference turns:
    - Minimum duration (default 1.2s) — short fragments are unreliable.
    - Audio must be available.
    - No overlap with other speakers.
    - Minimum quality score (ASR confidence, voiced ratio, etc.).
    - Do NOT use text emotion labels — baseline is purely acoustic.

    The baseline is built once and frozen. It is NOT updated with later turns.

    Args:
        min_turn_duration_s: Minimum turn duration to be a baseline candidate.
        min_reference_turns: Minimum reference turns needed for a valid baseline.
        min_reference_seconds: Minimum total reference duration.
        min_quality_score: Minimum quality score for a turn to be used.
    """

    def __init__(
        self,
        min_turn_duration_s: float = 1.2,
        min_reference_turns: int = 3,
        min_reference_seconds: float = 5.0,
        min_quality_score: float = 0.60,
    ):
        self.min_turn_duration_s = min_turn_duration_s
        self.min_reference_turns = min_reference_turns
        self.min_reference_seconds = min_reference_seconds
        self.min_quality_score = min_quality_score

    def build(
        self,
        speaker: str,
        turn_records: List[Dict[str, Any]],
    ) -> SpeakerBaselineProfile:
        """
        Build baseline from candidate early-call turns.

        Args:
            speaker: Speaker identifier (e.g., "Agent", "Customer").
            turn_records: List of dicts with keys:
                - turn_id (int)
                - duration (float)
                - prosody_vector (np.ndarray) — extracted acoustic features
                - wavlm_embedding (np.ndarray) — WavLM pooled embedding
                - audio_available (bool)
                - overlap (bool) — whether this turn overlaps with another speaker
                - quality_score (float) — 0-1 quality estimate

        Returns:
            SpeakerBaselineProfile (may be not ready if insufficient data).
        """
        candidates = [
            t for t in turn_records
            if t.get("duration", 0) >= self.min_turn_duration_s
            and t.get("audio_available", False)
            and not t.get("overlap", False)
            and t.get("quality_score", 0.0) >= self.min_quality_score
            and t.get("prosody_vector") is not None
            and t.get("wavlm_embedding") is not None
        ]

        if not candidates:
            return self._empty(speaker)

        # Select earliest qualifying turns (chronological order)
        total_s = 0.0
        selected: List[Dict[str, Any]] = []

        for turn in candidates:
            selected.append(turn)
            total_s += turn["duration"]

            if (
                len(selected) >= self.min_reference_turns
                and total_s >= self.min_reference_seconds
            ):
                break

        if len(selected) < self.min_reference_turns:
            return self._empty(speaker)

        # Stack features
        features = np.stack([t["prosody_vector"] for t in selected])
        embeddings = np.stack([t["wavlm_embedding"] for t in selected])

        # Robust statistics: median + MAD (not mean + std)
        f_center, f_scale = self._robust_stats(features)
        e_center, e_scale = self._robust_stats(embeddings)

        return SpeakerBaselineProfile(
            speaker=speaker,
            ready=True,
            n_reference_turns=len(selected),
            n_reference_seconds=total_s,
            feature_center=f_center,
            feature_scale=f_scale,
            embedding_center=e_center,
            embedding_scale=e_scale,
            reference_turn_ids=[t["turn_id"] for t in selected],
        )

    @staticmethod
    def _robust_stats(x: np.ndarray):
        """
        Robust center and scale using median and MAD.
        1.4826 * MAD ≈ std for normal distributions.
        Floor at 1e-4 to prevent division by zero.
        """
        center = np.median(x, axis=0)
        mad = np.median(np.abs(x - center), axis=0)
        scale = np.maximum(1.4826 * mad, 1e-4)
        return center.astype(np.float32), scale.astype(np.float32)

    @staticmethod
    def _empty(speaker: str) -> SpeakerBaselineProfile:
        return SpeakerBaselineProfile(
            speaker=speaker,
            ready=False,
            n_reference_turns=0,
            n_reference_seconds=0.0,
            feature_center=np.empty(0, dtype=np.float32),
            feature_scale=np.empty(0, dtype=np.float32),
            embedding_center=np.empty(0, dtype=np.float32),
            embedding_scale=np.empty(0, dtype=np.float32),
            reference_turn_ids=[],
        )
