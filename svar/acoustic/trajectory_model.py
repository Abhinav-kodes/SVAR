"""
Speaker-shift temporal model for within-call acoustic trajectory analysis.

Takes the sequence of baseline-relative acoustic vectors for a single
speaker within a call and predicts per-turn arousal, voice shift, and
escalation. This model learns from real call labels, NOT from EmoInHindi.
"""
from __future__ import annotations
from typing import Dict, Optional, Any
import torch
import torch.nn as nn


class SpeakerShiftTemporalModel(nn.Module):
    """
    BiGRU temporal model over a speaker's ordered turns within a call.

    Predicts:
    - arousal: scalar, how activated the speaker sounds (0-1)
    - voice_shift: scalar, how different from their own baseline (0-1)
    - escalation: scalar, whether negative activation is rising/sustained (0-1)

    These are trained on REAL call data with human-annotated labels.
    EmoInHindi does not provide these labels.

    Args:
        input_dim: Dimension of per-turn input features (relative prosody + embedding + delta).
        hidden_dim: GRU hidden dimension.
        num_layers: Number of GRU layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 515,  # ~32 prosody + 768 WavLM + 3 scalars
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Output heads (each predicts a scalar per turn)
        self.arousal_head = nn.Linear(hidden_dim * 2, 1)
        self.shift_head = nn.Linear(hidden_dim * 2, 1)
        self.escalation_head = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass over one speaker's ordered turns.

        Args:
            x: (B, T, input_dim) sequence of per-turn features.
            mask: (B, T) 1 for valid turns, 0 for padding.

        Returns:
            Dict with:
            - arousal: (B, T) arousal score per turn
            - voice_shift: (B, T) voice shift score per turn
            - escalation: (B, T) escalation score per turn
            - hidden: (B, T, hidden_dim*2) raw hidden states
        """
        projected = self.input_proj(x)

        if mask is not None:
            lengths = mask.sum(dim=1).cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                projected, lengths, batch_first=True, enforce_sorted=False,
            )
            gru_out, _ = self.gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out, batch_first=True, total_length=x.shape[1],
            )
        else:
            gru_out, _ = self.gru(projected)

        gru_out = self.dropout(gru_out)

        arousal = torch.sigmoid(self.arousal_head(gru_out)).squeeze(-1)
        voice_shift = torch.sigmoid(self.shift_head(gru_out)).squeeze(-1)
        escalation = torch.sigmoid(self.escalation_head(gru_out)).squeeze(-1)

        return {
            "arousal": arousal,
            "voice_shift": voice_shift,
            "escalation": escalation,
            "hidden": gru_out,
        }

    @staticmethod
    def compute_loss(
        outputs: Dict[str, torch.Tensor],
        arousal_targets: Optional[torch.Tensor] = None,
        shift_targets: Optional[torch.Tensor] = None,
        escalation_targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute training loss with masking.

        Targets use IGNORE_INDEX (-100) for turns without labels.
        """
        total_loss = torch.tensor(0.0, device=next(iter(outputs.values())).device)
        n_heads = 0

        if arousal_targets is not None:
            valid = (arousal_targets != -100) & (mask if mask is not None else torch.ones_like(arousal_targets, dtype=torch.bool))
            if valid.any():
                loss = nn.functional.binary_cross_entropy(
                    outputs["arousal"][valid],
                    arousal_targets[valid].float(),
                )
                total_loss = total_loss + loss
                n_heads += 1

        if shift_targets is not None:
            valid = (shift_targets != -100) & (mask if mask is not None else torch.ones_like(shift_targets, dtype=torch.bool))
            if valid.any():
                loss = nn.functional.binary_cross_entropy(
                    outputs["voice_shift"][valid],
                    shift_targets[valid].float(),
                )
                total_loss = total_loss + loss
                n_heads += 1

        if escalation_targets is not None:
            valid = (escalation_targets != -100) & (mask if mask is not None else torch.ones_like(escalation_targets, dtype=torch.bool))
            if valid.any():
                loss = nn.functional.binary_cross_entropy(
                    outputs["escalation"][valid],
                    escalation_targets[valid].float(),
                )
                total_loss = total_loss + loss
                n_heads += 1

        if n_heads > 0:
            total_loss = total_loss / n_heads

        return total_loss
