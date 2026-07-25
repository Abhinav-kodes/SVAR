"""
Data schemas for the SVAR emotion recognition pipeline.

All prediction outputs use dataclasses for type safety and serialization.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import numpy as np


# ── Emotion taxonomy (EmoInHindi canonical 16-label) ──

EMOINHINDI_EMOTIONS = [
    "neutral",
    "anticipation",
    "anger",
    "sad",
    "confident",
    "fear",
    "disgusted",
    "surprised",
    "hopeful",
    "annoyed",
    "compassion",
    "joy",
    "apprehensive",
    "grateful",
    "guilty",
    "impressed",
]

# UI-friendly display names — applied ONLY after inference, never during training
EMOTION_DISPLAY_NAMES = {
    "joy": "happy",
    "surprised": "surprise",
    "sad": "sad",
}

# Sentiment derived from emotion (for dataset prep only — not a model output target)
EMOTION_TO_SENTIMENT = {
    "neutral": "neutral",
    "anticipation": "neutral",
    "anger": "negative",
    "sad": "negative",
    "confident": "positive",
    "fear": "negative",
    "disgusted": "negative",
    "surprised": "neutral",
    "hopeful": "positive",
    "annoyed": "negative",
    "compassion": "positive",
    "joy": "positive",
    "apprehensive": "negative",
    "grateful": "positive",
    "guilty": "negative",
    "impressed": "positive",
}

# These task heads exist ONLY for real call data — not EmoInHindi
INTERACTION_STATES = ["calm", "tension", "escalation", "peak_conflict", "recovery"]
CONDUCT_RISKS = ["insult_or_degradation", "profanity", "intimidation_or_threat", "harassment"]

SENTIMENTS = ["negative", "neutral", "positive"]

IGNORE_INDEX = -100


# ── Input schemas ──

@dataclass
class DiarizedSegment:
    """A single diarized ASR segment from the pipeline."""
    id: int
    speaker: str
    start: float
    end: float
    text: str
    audio_path: Optional[str] = None
    asr_confidence: Optional[float] = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class AnalysisTurn:
    """A repaired/merged turn ready for emotion analysis (inference only)."""
    turn_id: int
    speaker: str
    start: float
    end: float
    duration: float
    text: str
    segment_ids: List[int]
    too_short: bool
    asr_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContextInput:
    """Contextual input for the text model."""
    turn_id: int
    speaker: str
    context_text: str
    target_text: str
    full_sequence: str
    segment_ids: List[int]


# ── Acoustic schemas ──

@dataclass
class SpeakerBaselineProfile:
    """Per-speaker baseline built from early-call reference turns."""
    speaker: str
    ready: bool
    n_reference_turns: int
    n_reference_seconds: float
    feature_center: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    feature_scale: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    embedding_center: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    embedding_scale: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    reference_turn_ids: List[int] = field(default_factory=list)


@dataclass
class AcousticOutput:
    """Per-turn acoustic analysis output."""
    arousal: Optional[float] = None
    valence: Optional[float] = None
    voice_shift: Optional[float] = None
    escalation: Optional[float] = None
    relative_features: Optional[np.ndarray] = None
    baseline_ready: bool = False
    quality_score: float = 0.0
    audio_available: bool = False
    acoustic_status: str = "no_audio"  # no_audio | baseline_warming_up | ready | error


# ── Output schemas ──

@dataclass
class EmotionOutput:
    """Multi-label emotion prediction with probabilities."""
    probabilities: Dict[str, float]
    primary: str = ""
    intensities: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.primary and self.probabilities:
            self.primary = max(self.probabilities, key=self.probabilities.get)


@dataclass
class SentimentOutput:
    label: str
    confidence: float


@dataclass
class InteractionStateOutput:
    label: str
    confidence: float


@dataclass
class ConductRiskOutput:
    probabilities: Dict[str, float]

    @property
    def active_risks(self) -> Dict[str, float]:
        return {k: v for k, v in self.probabilities.items() if v > 0.5}


@dataclass
class UncertaintyOutput:
    label: str  # "confident" | "uncertain" | "insufficient_evidence"
    score: float
    evidence_quality: str = "good"  # good | short_text | short_audio | low_confidence | high_entropy


@dataclass
class ModalityInfo:
    used_text: bool
    used_audio: bool
    fusion_mode: str  # "text_only" | "audio_only" | "learned_fusion" | "unavailable"


@dataclass
class TurnPrediction:
    """Complete prediction for a single analysis turn."""
    turn_id: int
    speaker: str
    start: float
    end: float
    text: str
    segment_ids: List[int]
    emotion: EmotionOutput
    sentiment: SentimentOutput
    interaction_state: InteractionStateOutput
    conduct_risk: ConductRiskOutput
    uncertainty: UncertaintyOutput
    acoustic: AcousticOutput
    modality: ModalityInfo
    model_version: str = "2.0.0"
    inference_mode: str = "text_only"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert numpy arrays to lists for serialization
        if self.acoustic.relative_features is not None:
            d["acoustic"]["relative_features"] = self.acoustic.relative_features.tolist()
        return d

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Backwards-compatible dict for existing downstream modules."""
        return {
            "emotion": self.emotion.primary,
            "sentiment": self.sentiment.label,
            "confidence": self.sentiment.confidence,
            "fusion_source": self.modality.fusion_mode,
            "emotion_probabilities": self.emotion.probabilities,
            "interaction_state": self.interaction_state.label,
            "conduct_risk": self.conduct_risk.probabilities,
            "uncertainty": self.uncertainty.label,
            "acoustic_arousal": self.acoustic.arousal,
            "acoustic_valence": self.acoustic.valence,
            "voice_shift": self.acoustic.voice_shift,
            "escalation": self.acoustic.escalation,
        }
