"""
Temperature scaling calibration + evidence quality gating.

Post-hoc calibration of model probabilities using a learned temperature
parameter per task head.

Separates two concerns:
1. Evidence quality gating: deterministic checks on data completeness
   (short text, short audio, etc.) — these are diagnostics, not predictions.
2. Learned uncertainty: the model's own uncertainty head determines
   semantic confidence. Temperature scaling adjusts overconfidence.

When evidence is insufficient, the system outputs "uncertain", never "neutral".
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class CalibrationResult:
    """Calibrated prediction for a single turn."""
    emotion: Dict[str, float]
    sentiment: Dict[str, float]
    interaction_state: Dict[str, float]
    conduct_risk: Dict[str, float]

    # Evidence quality diagnostics (deterministic, not predictions)
    evidence_quality: str = "good"  # good | short_text | short_audio | low_confidence | high_entropy

    # Entropy of emotion distribution (high = uncertain)
    emotion_entropy: float = 0.0

    # Dominant emotion margin (top1 - top2, low = ambiguous)
    emotion_margin: float = 0.0


class TemperatureScaler(nn.Module):
    """
    Per-head temperature scaling for probability calibration.

    Learns a single scalar temperature T per task head.
    T > 1 flattens distribution (less confident).
    T < 1 sharpens distribution (more confident).

    Args:
        n_heads: Number of task heads to calibrate.
    """

    def __init__(self, n_heads: int = 4):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(n_heads))

    def calibrate(
        self,
        emotion_logits: torch.Tensor,
        sentiment_logits: torch.Tensor,
        interaction_state_logits: torch.Tensor,
        conduct_risk_logits: torch.Tensor,
    ) -> CalibrationResult:
        """
        Apply temperature scaling to raw logits.

        Returns calibrated probabilities. Abstention/evidence quality
        is determined separately by EvidenceQualityGate.
        """
        with torch.no_grad():
            emotion_scaled = emotion_logits / self.temperatures[0]
            sentiment_scaled = sentiment_logits / self.temperatures[1]
            is_scaled = interaction_state_logits / self.temperatures[2]
            cr_scaled = conduct_risk_logits / self.temperatures[3]

            emotion_probs = torch.sigmoid(emotion_scaled)
            sentiment_probs = F.softmax(sentiment_scaled, dim=-1)
            is_probs = F.softmax(is_scaled, dim=-1)
            cr_probs = torch.sigmoid(cr_scaled)

            emotion_top2 = torch.topk(emotion_probs, min(2, len(emotion_probs)))
            emotion_margin = (
                emotion_top2.values[0] - emotion_top2.values[1]
            ).item() if len(emotion_top2.values) >= 2 else 1.0

            emotion_clamped = emotion_probs.clamp(1e-7, 1 - 1e-7)
            entropy = -(
                emotion_clamped * torch.log(emotion_clamped)
                + (1 - emotion_clamped) * torch.log(1 - emotion_clamped)
            ).sum().item()
            max_entropy = len(emotion_probs) * np.log(2)
            emotion_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            from .schemas import EMOINHINDI_EMOTIONS, SENTIMENTS, INTERACTION_STATES, CONDUCT_RISKS

            emotion_dict = {
                EMOINHINDI_EMOTIONS[i]: round(emotion_probs[i].item(), 6)
                for i in range(min(len(EMOINHINDI_EMOTIONS), len(emotion_probs)))
            }
            sentiment_dict = {
                SENTIMENTS[i]: round(sentiment_probs[i].item(), 6)
                for i in range(min(len(SENTIMENTS), len(sentiment_probs)))
            }
            is_dict = {
                INTERACTION_STATES[i]: round(is_probs[i].item(), 6)
                for i in range(min(len(INTERACTION_STATES), len(is_probs)))
            }
            cr_dict = {
                CONDUCT_RISKS[i]: round(cr_probs[i].item(), 6)
                for i in range(min(len(CONDUCT_RISKS), len(cr_probs)))
            }

        return CalibrationResult(
            emotion=emotion_dict,
            sentiment=sentiment_dict,
            interaction_state=is_dict,
            conduct_risk=cr_dict,
            emotion_entropy=emotion_entropy,
            emotion_margin=emotion_margin,
        )

    def fit_temperature(
        self,
        val_logits: Dict[str, torch.Tensor],
        val_labels: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Learn optimal temperatures on validation set (NLL minimization).
        """
        best_temps = {}
        head_names = ["emotion", "sentiment", "interaction_state", "conduct_risk"]

        for idx, name in enumerate(head_names):
            if name not in val_logits:
                continue
            logits = val_logits[name]
            labels = val_labels[name]

            best_nll = float("inf")
            best_t = 1.0

            for t in np.arange(0.5, 3.0, 0.1):
                scaled = logits / t
                if name in ("emotion", "conduct_risk"):
                    probs = torch.sigmoid(scaled)
                    nll = F.binary_cross_entropy(probs, labels.float()).item()
                else:
                    nll = F.cross_entropy(scaled, labels).item()

                if nll < best_nll:
                    best_nll = nll
                    best_t = t

            best_temps[name] = round(best_t, 3)
            with torch.no_grad():
                self.temperatures[idx] = best_t

        return best_temps


class EvidenceQualityGate:
    """
    Deterministic data-quality checks.

    These are diagnostics about input completeness, NOT emotion predictions.
    When evidence is poor, output "uncertain" — never map to "neutral".

    Returns an evidence quality string that the pipeline uses to decide
    whether to trust the model's output or label it uncertain.
    """

    def __init__(
        self,
        min_words: int = 2,
        min_duration_s: float = 0.5,
        min_confidence: float = 0.3,
    ):
        self.min_words = min_words
        self.min_duration_s = min_duration_s
        self.min_confidence = min_confidence

    def assess(
        self,
        text: str,
        duration_s: float,
        asr_confidence: Optional[float] = None,
        emotion_entropy: float = 0.0,
        emotion_margin: float = 1.0,
    ) -> str:
        """
        Assess evidence quality. Returns one of:
        - "good": evidence is sufficient
        - "short_text": text too short for semantic analysis
        - "short_audio": audio segment too brief
        - "low_asr_confidence": ASR transcript unreliable
        - "high_entropy": model distribution is flat (ambiguous)
        - "low_margin": top emotions are too close to distinguish
        """
        if not text or len(text.split()) < self.min_words:
            return "short_text"

        if duration_s < self.min_duration_s:
            return "short_audio"

        if asr_confidence is not None and asr_confidence < self.min_confidence:
            return "low_asr_confidence"

        if emotion_entropy > 0.8:
            return "high_entropy"

        if emotion_margin < 0.05:
            return "low_margin"

        return "good"
