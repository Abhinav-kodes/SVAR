"""
Context construction for contextual emotion classification.

Builds role-tagged context windows around target turns for MuRIL input.
"""
from __future__ import annotations
from typing import List, Optional
from .schemas import AnalysisTurn, ContextInput


_ROLE_TOKENS = {
    "agent": "[AGENT]",
    "customer": "[CUSTOMER]",
}
_TARGET_ROLE_TOKENS = {
    "agent": "[TARGET_AGENT]",
    "customer": "[TARGET_CUSTOMER]",
}


def _normalize_speaker(speaker: str) -> str:
    s = speaker.strip().lower()
    if s in ("agent", "agt"):
        return "agent"
    return "customer"


def _role_token(speaker: str, is_target: bool = False) -> str:
    key = _normalize_speaker(speaker)
    if is_target:
        return _TARGET_ROLE_TOKENS[key]
    return _ROLE_TOKENS[key]


def build_contexts(
    turns: List[AnalysisTurn],
    left_context: int = 2,
    right_context: int = 1,
    max_length: int = 256,
) -> List[ContextInput]:
    """
    Build contextual inputs for each turn using neighboring turns.

    Format:
        [CLS] [AGENT] prev2 [CUSTOMER] prev1 [TARGET_X] target [AGENT] next1 [SEP]

    The target turn is always preserved in full. Context is truncated if needed.

    Args:
        turns: Ordered list of repaired analysis turns.
        left_context: Number of preceding turns to include.
        right_context: Number of following turns to include.
        max_length: Maximum token count for the full sequence (approximate by word count).

    Returns:
        List of ContextInput, one per turn.
    """
    contexts: List[ContextInput] = []
    n = len(turns)

    for i, turn in enumerate(turns):
        context_parts: List[str] = []
        target_idx = i
        target_token = _role_token(turn.speaker, is_target=True)
        target_part = f"{target_token} {turn.text}"

        # Left context (nearest first)
        left_start = max(0, i - left_context)
        for j in range(left_start, i):
            t = turns[j]
            context_parts.append(f"{_role_token(t.speaker)} {t.text}")

        # Target
        context_parts.append(target_part)

        # Right context
        right_end = min(n, i + right_context + 1)
        for j in range(i + 1, right_end):
            t = turns[j]
            context_parts.append(f"{_role_token(t.speaker)} {t.text}")

        full_sequence = " ".join(context_parts)

        # Truncate if needed: preserve target, then nearest context
        words = full_sequence.split()
        if len(words) > max_length:
            # Always keep target
            target_words = target_part.split()
            budget = max_length - len(target_words)
            if budget <= 0:
                full_sequence = target_part
            else:
                left_parts = []
                left_budget = budget // 2
                right_parts = []
                right_budget = budget - left_budget

                # Rebuild left
                for j in range(left_start, i):
                    t = turns[j]
                    part = f"{_role_token(t.speaker)} {t.text}".split()
                    if len(part) <= left_budget:
                        left_parts.extend(part)
                        left_budget -= len(part)
                    else:
                        break

                # Rebuild right
                for j in range(i + 1, right_end):
                    t = turns[j]
                    part = f"{_role_token(t.speaker)} {t.text}".split()
                    if len(part) <= right_budget:
                        right_parts.extend(part)
                        right_budget -= len(part)
                    else:
                        break

                full_sequence = " ".join(
                    left_parts + target_words + right_parts
                )

        contexts.append(ContextInput(
            turn_id=turn.turn_id,
            speaker=turn.speaker,
            context_text=full_sequence,
            target_text=turn.text,
            full_sequence=full_sequence,
            segment_ids=turn.segment_ids,
        ))

    return contexts
