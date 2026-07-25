"""
Transcript-based speaker role classifier.

Uses MuRIL (Multilingual Representations for Indian Languages) to predict
which diarized speaker is the agent and which is the customer based on
transcript semantics, not voice identity.

Output classes:
  0: A_AGENT_B_CUSTOMER  (Speaker A is agent)
  1: A_CUSTOMER_B_AGENT  (Speaker B is agent)
  2: UNKNOWN_OR_OTHER     (not enough evidence / not a service call)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


Role = Literal["agent", "customer", "unknown"]

ROLE_MAPPINGS = [
    "A_AGENT_B_CUSTOMER",
    "A_CUSTOMER_B_AGENT",
    "UNKNOWN_OR_OTHER",
]


@dataclass
class SpeakerRoleResult:
    speaker_a: Role
    speaker_b: Role
    confidence: float
    mapping: str
    status: str  # "resolved" | "uncertain"
    turns_used: int


class SpeakerRoleClassifier(nn.Module):
    """MuRIL-based binary role classifier over speaker-tagged transcript context.

    Input:  [SPK_A] helo ... [SPK_B] haan ... [SPK_A] ...
    Output: 3-class logits (A_AGENT_B_CUSTOMER, A_CUSTOMER_B_AGENT, UNKNOWN)
    """

    def __init__(
        self,
        model_name: str = "google/muril-base-cased",
        num_classes: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls = out.last_hidden_state[:, 0]
        return self.head(cls)


def build_role_context(
    turns: list,
    max_turns: int = 12,
    min_words: int = 2,
) -> str:
    """Build a speaker-tagged transcript context from diarized turns.

    Uses [SPK_A] and [SPK_B] tokens — never pre-assigned agent/customer labels.
    Filters out short/empty fragments. Takes the earliest meaningful turns.
    """
    parts = []
    seen_speakers = {}

    for turn in turns:
        text = ""
        if hasattr(turn, "text"):
            text = (turn.text or "").strip()
        elif isinstance(turn, dict):
            text = (turn.get("text", "") or "").strip()

        if len(text.split()) < min_words:
            continue

        speaker = ""
        if hasattr(turn, "speaker"):
            speaker = str(turn.speaker).strip()
        elif isinstance(turn, dict):
            speaker = str(turn.get("speaker", "")).strip()

        speaker_lower = speaker.lower()
        if speaker_lower not in seen_speakers:
            seen_speakers[speaker_lower] = len(seen_speakers)
        spk_idx = seen_speakers[speaker_lower]
        token = f"[SPK_{chr(65 + spk_idx)}]"

        parts.append(f"{token} {text}")

        if len(parts) >= max_turns:
            break

    return " ".join(parts)


@torch.no_grad()
def infer_roles(
    model: SpeakerRoleClassifier,
    tokenizer,
    diarized_turns: list,
    device: str = "cuda",
    max_turns: int = 12,
    min_confidence: float = 0.70,
    min_lock_confidence: float = 0.85,
    min_lock_turns: int = 6,
) -> SpeakerRoleResult:
    """Predict the role mapping for a two-speaker call.

    Args:
        model: Trained SpeakerRoleClassifier.
        tokenizer: Matching MuRIL tokenizer.
        diarized_turns: List of turns (dicts or dataclasses with .text, .speaker).
        device: Compute device.
        max_turns: Max transcript turns to include.
        min_confidence: Below this, return UNKNOWN.
        min_lock_confidence: Above this + min_lock_turns, status = "resolved".
        min_lock_turns: Min turns needed for a resolved status.

    Returns:
        SpeakerRoleResult with speaker_a, speaker_b, confidence, mapping, status.
    """
    model.eval()
    text = build_role_context(diarized_turns, max_turns=max_turns)

    if not text.strip():
        return SpeakerRoleResult(
            speaker_a="unknown",
            speaker_b="unknown",
            confidence=1.0,
            mapping="UNKNOWN_OR_OTHER",
            status="uncertain",
            turns_used=0,
        )

    encoded = tokenizer(
        text,
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    logits = model(**encoded)
    probs = torch.softmax(logits[0], dim=-1)

    idx = int(probs.argmax())
    conf = float(probs[idx])

    if conf < min_confidence:
        return SpeakerRoleResult(
            speaker_a="unknown",
            speaker_b="unknown",
            confidence=conf,
            mapping="UNKNOWN_OR_OTHER",
            status="uncertain",
            turns_used=min(max_turns, len(diarized_turns)),
        )

    role_a, role_b, mapping = _decode_mapping(idx)

    n_turns = min(max_turns, len(diarized_turns))
    status = "resolved" if (conf >= min_lock_confidence and n_turns >= min_lock_turns) else "provisional"

    return SpeakerRoleResult(
        speaker_a=role_a,
        speaker_b=role_b,
        confidence=conf,
        mapping=mapping,
        status=status,
        turns_used=n_turns,
    )


def _decode_mapping(idx: int):
    if idx == 0:
        return "agent", "customer", "A_AGENT_B_CUSTOMER"
    elif idx == 1:
        return "customer", "agent", "A_CUSTOMER_B_AGENT"
    else:
        return "unknown", "unknown", "UNKNOWN_OR_OTHER"


def apply_role_mapping(segments: list, result: SpeakerRoleResult) -> dict:
    """Apply a SpeakerRoleResult to a list of segments, returning role_mapping.

    Does NOT mutate segments. Returns a dict mapping spk_0/spk_1 → agent/customer.

    The caller should use this mapping to update segments after role resolution.
    """
    spk_speakers = sorted(set(
        seg.get("speaker", "") for seg in segments
        if seg.get("speaker", "").startswith("spk_")
    ))

    mapping = {}

    if len(spk_speakers) < 2 or result.status == "uncertain":
        for spk in spk_speakers:
            mapping[spk] = "unknown"
        return mapping

    if len(spk_speakers) >= 2:
        spk_a = spk_speakers[0]
        spk_b = spk_speakers[1]

        if result.mapping == "A_AGENT_B_CUSTOMER":
            mapping[spk_a] = "agent"
            mapping[spk_b] = "customer"
        elif result.mapping == "A_CUSTOMER_B_AGENT":
            mapping[spk_a] = "customer"
            mapping[spk_b] = "agent"
        else:
            mapping[spk_a] = "unknown"
            mapping[spk_b] = "unknown"

    return mapping
