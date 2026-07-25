"""
WavLM-based audio emotion encoder.

Extracts call-style acoustic emotion features from raw waveform segments.
Uses microsoft/wavlm-base-plus with gradient checkpointing for VRAM safety.
"""
from __future__ import annotations
from typing import Dict, Optional, Any, Literal
import torch
import torch.nn as nn


class WavLMEmotionModel(nn.Module):
    """
    WavLM encoder with emotion head for acoustic emotion from raw audio.

    Design:
    - WavLM base (768-dim hidden) as frozen or fine-tuned feature extractor.
    - Aggregates frame-level hidden states via attention pooling.
    - Multi-task heads: emotion (16), intensity (16), interaction_state (5).
    - Gradient checkpointing always enabled for VRAM safety.

    Args:
        model_name: HuggingFace WavLM model identifier.
        num_emotions: Number of emotion classes.
        num_interaction_states: Number of interaction states.
        dropout: Dropout rate.
        freeze_base: If True, freeze WavLM encoder parameters.
        max_audio_length_s: Maximum audio segment length in seconds.
        sample_rate: Expected sample rate (16000 for WavLM).
        pooling: Aggregation strategy: "attention" or "mean".
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        num_emotions: int = 16,
        num_interaction_states: int = 5,
        dropout: float = 0.20,
        freeze_base: bool = True,
        max_audio_length_s: float = 30.0,
        sample_rate: int = 16000,
        pooling: Literal["attention", "mean"] = "attention",
    ):
        super().__init__()
        self.config = {
            "model_name": model_name,
            "num_emotions": num_emotions,
            "num_interaction_states": num_interaction_states,
            "dropout": dropout,
            "freeze_base": freeze_base,
            "max_audio_length_s": max_audio_length_s,
            "sample_rate": sample_rate,
            "pooling": pooling,
        }
        self.sample_rate = sample_rate
        self.pooling = pooling
        self.max_length = int(max_audio_length_s * sample_rate)

        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(
            model_name,
            gradient_checkpointing=True,
        )
        hidden_size = self.encoder.config.hidden_size  # 768

        if freeze_base:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if pooling == "attention":
            self.attn_pool = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )

        self.dropout = nn.Dropout(dropout)

        # Multi-task heads (smaller than text model to keep VRAM down)
        self.emotion_head = nn.Linear(hidden_size, num_emotions)
        self.intensity_head = nn.Linear(hidden_size, num_emotions)
        self.interaction_state_head = nn.Linear(hidden_size, num_interaction_states)

    def _pool(
        self, hidden: torch.Tensor, mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool frame-level hidden states to a single vector per sample."""
        if self.pooling == "mean":
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = hidden.mean(dim=1)
            return pooled

        # Attention pooling
        attn_scores = self.attn_pool(hidden).squeeze(-1)  # (B, T)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        pooled = (hidden * attn_weights).sum(dim=1)  # (B, H)
        return pooled

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            input_values: (B, L) raw waveform float tensor at self.sample_rate.
            attention_mask: (B, L) 1 for valid samples, 0 for padding.

        Returns:
            Dict with pooled embedding, emotion/intensity/is logits and probs.
        """
        # Truncate if too long (WavLM base handles ~30s max)
        if input_values.shape[1] > self.max_length:
            input_values = input_values[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]

        # Convert float attention mask to bool for WavLM
        wavlm_mask = attention_mask.bool() if attention_mask is not None else None

        with torch.amp.autocast("cuda", enabled=not self.training or True):
            outputs = self.encoder(
                input_values=input_values,
                attention_mask=wavlm_mask,
                output_hidden_states=False,
            )

        hidden = outputs.last_hidden_state  # (B, T, 768)
        pooled = self._pool(hidden, attention_mask)
        pooled = self.dropout(pooled)

        emotion_logits = self.emotion_head(pooled)
        intensity_raw = self.intensity_head(pooled)
        is_logits = self.interaction_state_head(pooled)

        return {
            "embedding": pooled,
            "emotion_logits": emotion_logits,
            "emotion_probs": torch.sigmoid(emotion_logits),
            "intensity": torch.sigmoid(intensity_raw),
            "interaction_state_logits": is_logits,
            "interaction_state_probs": torch.softmax(is_logits, dim=-1),
        }
