"""
Baseline-aware multimodal fusion network.

Fuses text (MuRIL CLS) and audio (speaker-relative acoustic features)
via learned MLP with modality dropout.

Key design principle: audio provides a CHANGE SIGNAL (arousal, voice shift,
escalation) that modulates the text-derived emotion, NOT competing emotion
labels. The text model determines WHAT emotion; the audio model determines
HOW INTENSE and whether it represents SUSTAINED ESCALATION.
"""
from __future__ import annotations
from typing import Dict, Optional, Any
import torch
import torch.nn as nn


class MultimodalFusionNet(nn.Module):
    """
    Baseline-aware text + audio fusion for emotion classification.

    Input streams:
    - Text: MuRIL CLS embedding (768-dim) → emotion/sentiment logits
    - Audio: Speaker-relative features (prosody_z + embedding_z + scalars)
             → arousal, voice_shift, escalation

    Fusion: The audio change signal modulates the text embedding via a
    gated fusion layer. The model learns when acoustic escalation should
    increase or decrease confidence in the text-derived emotional state.

    Args:
        text_dim: Dimension of text CLS embedding (768 for MuRIL).
        audio_relative_dim: Dimension of baseline-relative acoustic vector.
        arousal_dim: Dimension of arousal/shift/escalation features (3).
        fused_dim: Output fusion dimension.
        num_emotions: Number of emotion labels (16).
        num_sentiments: Number of sentiment classes (3).
        num_interaction_states: Number of interaction states (5).
        num_conduct_risks: Number of conduct risk labels (4).
        modality_dropout_p: Probability of dropping audio during training.
        dropout: General dropout rate.
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_relative_dim: int = 803,  # ~32 prosody + 768 WavLM + 3
        fused_dim: int = 512,
        num_emotions: int = 16,
        num_sentiments: int = 3,
        num_interaction_states: int = 5,
        num_conduct_risks: int = 4,
        modality_dropout_p: float = 0.30,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.config = {
            "text_dim": text_dim,
            "audio_relative_dim": audio_relative_dim,
            "fused_dim": fused_dim,
            "modality_dropout_p": modality_dropout_p,
        }

        self.modality_dropout_p = modality_dropout_p

        # Text stream: CLS embedding → emotion logits
        self.text_proj = nn.Linear(text_dim, fused_dim)

        # Audio stream: relative features → change signal
        # The audio stream does NOT predict emotion directly.
        # It predicts arousal, shift, escalation — scalar signals.
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_relative_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
        )

        # Change signal heads (audio-side outputs)
        self.arousal_head = nn.Linear(fused_dim, 1)
        self.shift_head = nn.Linear(fused_dim, 1)
        self.escalation_head = nn.Linear(fused_dim, 1)

        # Gated fusion: audio change signal modulates text
        self.gate = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.Sigmoid(),
        )

        # Confidence modulator: escalation signal adjusts emotion confidence
        self.confidence_modulator = nn.Sequential(
            nn.Linear(3, fused_dim),  # 3 = arousal + shift + escalation
            nn.Sigmoid(),
        )

        self.fused_norm = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(dropout)

        # Prediction heads on fused representation
        self.emotion_head = nn.Linear(fused_dim, num_emotions)
        self.sentiment_head = nn.Linear(fused_dim, num_sentiments)
        self.interaction_state_head = nn.Linear(fused_dim, num_interaction_states)
        self.conduct_risk_head = nn.Linear(fused_dim, num_conduct_risks)

    def forward(
        self,
        text_embedding: torch.Tensor,
        audio_relative: torch.Tensor,
        text_available: Optional[torch.Tensor] = None,
        audio_available: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            text_embedding: (B, text_dim) CLS from MuRIL.
            audio_relative: (B, audio_relative_dim) baseline-relative acoustic vector.
            text_available: (B,) bool, True if text embedding is real.
            audio_available: (B,) bool, True if audio is available and baseline is ready.

        Returns:
            Dict with fused_embedding, all logits/probs, and acoustic signals.
        """
        B = text_embedding.shape[0]
        device = text_embedding.device

        # Text stream
        text_emb = self.text_proj(text_embedding)

        # Audio stream (change signal only)
        audio_emb = self.audio_proj(audio_relative)
        arousal = torch.sigmoid(self.arousal_head(audio_emb)).squeeze(-1)
        voice_shift = torch.sigmoid(self.shift_head(audio_emb)).squeeze(-1)
        escalation = torch.sigmoid(self.escalation_head(audio_emb)).squeeze(-1)

        # Modality dropout during training
        if self.training and self.modality_dropout_p > 0:
            audio_drop = torch.bernoulli(
                torch.full((B, 1), self.modality_dropout_p, device=device),
            ).bool()
            audio_emb = audio_emb.masked_fill(audio_drop, 0.0)

        # Apply availability masks
        if text_available is not None:
            text_emb = text_emb * text_available.unsqueeze(-1).float()
        if audio_available is not None:
            audio_emb = audio_emb * audio_available.unsqueeze(-1).float()

        # Gated fusion: audio modulates text
        gate_input = torch.cat([text_emb, audio_emb], dim=-1)
        gate_values = self.gate(gate_input)
        fused = gate_values * text_emb + (1 - gate_values) * audio_emb

        # Confidence modulation: escalation signal adjusts the fused representation
        acoustic_signals = torch.stack([arousal, voice_shift, escalation], dim=-1)
        conf_mod = self.confidence_modulator(acoustic_signals)
        fused = fused * conf_mod

        fused = self.fused_norm(fused)
        fused = self.dropout(fused)

        # Prediction heads
        emotion_logits = self.emotion_head(fused)
        sentiment_logits = self.sentiment_head(fused)
        is_logits = self.interaction_state_head(fused)
        cr_logits = self.conduct_risk_head(fused)

        # Mode tracking
        if text_available is not None and audio_available is not None:
            both = text_available & audio_available
            text_only = text_available & ~audio_available
            neither = ~text_available & ~audio_available
        else:
            both = torch.ones(B, dtype=torch.bool, device=device)
            text_only = torch.zeros(B, dtype=torch.bool, device=device)
            neither = torch.zeros(B, dtype=torch.bool, device=device)

        return {
            "fused_embedding": fused,
            "gate_values": gate_values,
            # Acoustic change signals
            "arousal": arousal,
            "voice_shift": voice_shift,
            "escalation": escalation,
            # Emotion predictions (fused text + audio)
            "emotion_logits": emotion_logits,
            "emotion_probs": torch.sigmoid(emotion_logits),
            "sentiment_logits": sentiment_logits,
            "sentiment_probs": torch.softmax(sentiment_logits, dim=-1),
            "interaction_state_logits": is_logits,
            "interaction_state_probs": torch.softmax(is_logits, dim=-1),
            "conduct_risk_logits": cr_logits,
            "conduct_risk_probs": torch.sigmoid(cr_logits),
            # Availability
            "both_available": both,
            "text_only": text_only,
            "neither_available": neither,
        }
