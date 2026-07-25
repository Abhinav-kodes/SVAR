"""
Temporal sequence decoder for call-level emotion analysis.

Takes ordered per-turn embeddings from a single call and produces
revised per-turn predictions using temporal context.
"""
from __future__ import annotations
from typing import Dict, Optional, Any, Literal
import torch
import torch.nn as nn


class CallTemporalDecoder(nn.Module):
    """
    BiGRU (or Transformer) temporal decoder over turn embeddings.

    Args:
        input_dim: Dimension of per-turn embeddings from the text/audio model.
        hidden_dim: GRU hidden dimension.
        num_layers: Number of GRU layers.
        num_emotions: Number of emotion labels.
        num_interaction_states: Number of interaction states.
        num_conduct_risks: Number of conduct risk labels.
        dropout: Dropout rate.
        mode: "bigru" or "transformer".
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_emotions: int = 16,
        num_interaction_states: int = 5,
        num_conduct_risks: int = 5,
        dropout: float = 0.20,
        mode: Literal["bigru", "transformer"] = "bigru",
    ):
        super().__init__()
        self.mode = mode
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if mode == "bigru":
            self.temporal = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            temporal_out_dim = hidden_dim * 2
        elif mode == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.temporal = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers,
            )
            temporal_out_dim = hidden_dim
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.dropout = nn.Dropout(dropout)

        # Revised prediction heads (smaller than base model)
        self.emotion_head = nn.Linear(temporal_out_dim, num_emotions)
        self.interaction_state_head = nn.Linear(temporal_out_dim, num_interaction_states)
        self.conduct_risk_head = nn.Linear(temporal_out_dim, num_conduct_risks)

    def forward(
        self,
        turn_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass over a sequence of turn embeddings.

        Args:
            turn_embeddings: (B, T, D) ordered turn embeddings.
            attention_mask: (B, T) 1 for valid turns, 0 for padding.

        Returns:
            Dict with revised logits and temporal embeddings.
        """
        B, T, _ = turn_embeddings.shape

        x = self.input_proj(turn_embeddings)

        if self.mode == "bigru":
            if attention_mask is not None:
                # Pack padded sequence
                lengths = attention_mask.sum(dim=1).cpu().clamp(min=1)
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False,
                )
                temporal_out, _ = self.temporal(packed)
                temporal_out, _ = nn.utils.rnn.pad_packed_sequence(
                    temporal_out, batch_first=True, total_length=T,
                )
            else:
                temporal_out, _ = self.temporal(x)
        else:
            # Transformer: use src_key_padding_mask
            src_key_padding_mask = None
            if attention_mask is not None:
                src_key_padding_mask = (attention_mask == 0)
            temporal_out = self.temporal(x, src_key_padding_mask=src_key_padding_mask)

        temporal_out = self.dropout(temporal_out)

        emotion_logits = self.emotion_head(temporal_out)
        is_logits = self.interaction_state_head(temporal_out)
        cr_logits = self.conduct_risk_head(temporal_out)

        return {
            "temporal_embeddings": temporal_out,
            "emotion_logits": emotion_logits,
            "emotion_probs": torch.sigmoid(emotion_logits),
            "interaction_state_logits": is_logits,
            "interaction_state_probs": torch.softmax(is_logits, dim=-1),
            "conduct_risk_logits": cr_logits,
            "conduct_risk_probs": torch.sigmoid(cr_logits),
        }
