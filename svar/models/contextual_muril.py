"""
Contextual MuRIL multi-task emotion model.

Base encoder: google/muril-base-cased
Heads: emotion (16-label multi-label), intensity (16), sentiment (3),
       interaction_state (5), conduct_risk (5 multi-label), uncertainty (1).
"""
from __future__ import annotations
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from torch.nn import functional as F


class ContextualCallEmotionModel(nn.Module):
    """
    MuRIL encoder with multi-task heads for call emotion analysis.

    Args:
        model_name: HuggingFace model identifier.
        num_emotions: Number of emotion labels (16 for EmoInHindi).
        num_sentiments: Number of sentiment classes (3).
        num_interaction_states: Number of interaction state classes (5).
        num_conduct_risks: Number of conduct risk labels (5).
        dropout: Dropout rate.
        freeze_base_layers: Number of bottom MuRIL layers to freeze.
    """

    def __init__(
        self,
        model_name: str = "google/muril-base-cased",
        num_emotions: int = 16,
        num_sentiments: int = 3,
        num_interaction_states: int = 5,
        num_conduct_risks: int = 5,
        dropout: float = 0.20,
        freeze_base_layers: int = 0,
    ):
        super().__init__()
        self.config = {
            "model_name": model_name,
            "num_emotions": num_emotions,
            "num_sentiments": num_sentiments,
            "num_interaction_states": num_interaction_states,
            "num_conduct_risks": num_conduct_risks,
            "dropout": dropout,
        }

        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for base

        if freeze_base_layers > 0:
            self._freeze_layers(freeze_base_layers)

        self.dropout = nn.Dropout(dropout)

        # Task heads
        self.emotion_head = nn.Linear(hidden_size, num_emotions)
        self.intensity_head = nn.Linear(hidden_size, num_emotions)
        self.sentiment_head = nn.Linear(hidden_size, num_sentiments)
        self.interaction_state_head = nn.Linear(hidden_size, num_interaction_states)
        self.conduct_risk_head = nn.Linear(hidden_size, num_conduct_risks)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _freeze_layers(self, n: int) -> None:
        """Freeze the bottom n encoder layers."""
        modules = list(self.encoder.encoder.layer[:n])
        for m in modules:
            for p in m.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        emotion_labels: Optional[torch.Tensor] = None,
        emotion_mask: Optional[torch.Tensor] = None,
        sentiment_labels: Optional[torch.Tensor] = None,
        interaction_state_labels: Optional[torch.Tensor] = None,
        conduct_risk_labels: Optional[torch.Tensor] = None,
        conduct_risk_mask: Optional[torch.Tensor] = None,
        intensity_labels: Optional[torch.Tensor] = None,
        intensity_mask: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            input_ids: (B, L) token IDs.
            attention_mask: (B, L) attention mask.
            emotion_labels: (B, 16) multi-label targets (-100 for missing).
            emotion_mask: (B, 16) 1 where label is valid, 0 where missing.
            sentiment_labels: (B,) targets (-100 for missing).
            interaction_state_labels: (B,) targets (-100 for missing).
            conduct_risk_labels: (B, 5) multi-label targets (-100 for missing).
            conduct_risk_mask: (B, 5) 1 where label is valid.
            intensity_labels: (B, 16) regression targets (-100 for missing).
            intensity_mask: (B, 16) 1 where intensity is valid.
            return_loss: If True, compute and return losses.

        Returns:
            Dict with logits, probabilities, cls_embedding, and optionally losses.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # CLS embedding
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (B, H)
        cls_emb = self.dropout(cls_emb)

        # Raw logits
        emotion_logits = self.emotion_head(cls_emb)       # (B, 16)
        intensity_raw = self.intensity_head(cls_emb)       # (B, 16)
        sentiment_logits = self.sentiment_head(cls_emb)    # (B, 3)
        is_logits = self.interaction_state_head(cls_emb)   # (B, 5)
        cr_logits = self.conduct_risk_head(cls_emb)        # (B, 5)
        unc_logit = self.uncertainty_head(cls_emb)         # (B, 1)

        # Probabilities
        emotion_probs = torch.sigmoid(emotion_logits)
        intensity_vals = torch.sigmoid(intensity_raw)  # normalized 0-1
        sentiment_probs = F.softmax(sentiment_logits, dim=-1)
        is_probs = F.softmax(is_logits, dim=-1)
        cr_probs = torch.sigmoid(cr_logits)
        unc_score = torch.sigmoid(unc_logit).squeeze(-1)

        result: Dict[str, Any] = {
            "cls_embedding": cls_emb,
            "emotion_logits": emotion_logits,
            "emotion_probs": emotion_probs,
            "intensity": intensity_vals,
            "sentiment_logits": sentiment_logits,
            "sentiment_probs": sentiment_probs,
            "interaction_state_logits": is_logits,
            "interaction_state_probs": is_probs,
            "conduct_risk_logits": cr_logits,
            "conduct_risk_probs": cr_probs,
            "uncertainty_logit": unc_logit.squeeze(-1),
            "uncertainty_score": unc_score,
        }

        if return_loss:
            result["losses"] = self._compute_losses(
                emotion_logits, intensity_raw, sentiment_logits,
                is_logits, cr_logits, unc_logit,
                emotion_labels, emotion_mask,
                sentiment_labels, interaction_state_labels,
                conduct_risk_labels, conduct_risk_mask,
                intensity_labels, intensity_mask,
            )
            if result["losses"]:
                result["total_loss"] = sum(result["losses"].values())
            else:
                result["total_loss"] = torch.tensor(0.0, device=emotion_logits.device)

        return result

    def _compute_losses(
        self,
        emotion_logits, intensity_raw, sentiment_logits,
        is_logits, cr_logits, unc_logit,
        emotion_labels, emotion_mask,
        sentiment_labels, interaction_state_labels,
        conduct_risk_labels, conduct_risk_mask,
        intensity_labels, intensity_mask,
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        # Emotion: multi-label BCE with masking
        if emotion_labels is not None and emotion_mask is not None:
            valid = emotion_mask.bool()
            loss = F.binary_cross_entropy_with_logits(
                emotion_logits[valid], emotion_labels[valid].float(),
                reduction="mean",
            )
            losses["emotion"] = loss

        # Intensity: SmoothL1 with masking
        if intensity_labels is not None and intensity_mask is not None:
            valid = intensity_mask.bool()
            if valid.any():
                losses["intensity"] = F.smooth_l1_loss(
                    intensity_raw[valid], intensity_labels[valid],
                )

        # Sentiment: cross-entropy
        if sentiment_labels is not None:
            valid = sentiment_labels != -100
            if valid.any():
                losses["sentiment"] = F.cross_entropy(
                    sentiment_logits[valid], sentiment_labels[valid],
                )

        # Interaction state: cross-entropy
        if interaction_state_labels is not None:
            valid = interaction_state_labels != -100
            if valid.any():
                losses["interaction_state"] = F.cross_entropy(
                    is_logits[valid], interaction_state_labels[valid],
                )

        # Conduct risk: multi-label BCE with masking
        if conduct_risk_labels is not None and conduct_risk_mask is not None:
            valid = conduct_risk_mask.bool()
            if valid.any():
                losses["conduct_risk"] = F.binary_cross_entropy_with_logits(
                    cr_logits[valid], conduct_risk_labels[valid].float(),
                    reduction="mean",
                )

        return losses
