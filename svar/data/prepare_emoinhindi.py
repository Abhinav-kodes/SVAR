"""
Prepare EmoInHindi dataset for training.

CRITICAL DESIGN RULES:
- Every original utterance is one supervised target. Do NOT merge/repair
  utterances during dataset preparation. Turn repair is for INFERENCE ONLY.
- Labels are carried by stable index, never matched by text.
- Interaction-state and conduct-risk use IGNORE_INDEX because EmoInHindi
  does not contain these labels. They are trained ONLY on real call data.
- Dialogue-level 70/15/15 split prevents data leakage.
"""
from __future__ import annotations
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from ..schemas import (
    EMOINHINDI_EMOTIONS, SENTIMENTS, INTERACTION_STATES, CONDUCT_RISKS,
    EMOTION_TO_SENTIMENT, IGNORE_INDEX,
)


def dialogue_split(
    data_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Load EmoInHindi and split at dialogue level.

    Expected input: JSONL files with fields per utterance:
        dialogue_id, utterance_id, speaker, start, end, text,
        emotion_label (list of strings), intensity (dict)

    Outputs per split:
        {split}_utterances.jsonl

    Returns:
        Dict with counts per split.
    """
    import random
    random.seed(seed)

    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load all utterances
    all_utterances: List[Dict[str, Any]] = []
    for f in sorted(data_path.glob("*.jsonl")):
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line.strip())
                all_utterances.append(row)

    for f in sorted(data_path.glob("*.csv")):
        with open(f, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                all_utterances.append(row)

    if not all_utterances:
        print(f"No data found in {data_path}. Creating example format.")
        _create_example_format(out_path)
        return {"train": 0, "val": 0, "test": 0}

    # Group by dialogue_id
    dialogues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for utt in all_utterances:
        did = utt.get("dialogue_id", utt.get("dialogueId", "unknown"))
        dialogues[did].append(utt)

    # Sort utterances within each dialogue by start time
    for did in dialogues:
        dialogues[did].sort(key=lambda x: float(x.get("start", 0)))

    # Stable index: tag each utterance with its position in its dialogue
    for did, utts in dialogues.items():
        for idx, utt in enumerate(utts):
            utt["_source_index"] = idx
            utt["_utterance_id"] = str(utt.get("utterance_id", utt.get("utterance_no", idx)))

    # Split dialogues (not utterances)
    dialogue_ids = list(dialogues.keys())
    random.shuffle(dialogue_ids)

    n = len(dialogue_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": dialogue_ids[:n_train],
        "val": dialogue_ids[n_train:n_train + n_val],
        "test": dialogue_ids[n_train + n_val:],
    }

    counts = {}
    for split_name, ids in splits.items():
        count = 0
        with open(out_path / f"{split_name}_utterances.jsonl", "w", encoding="utf-8") as fh:
            for did in ids:
                for utt in dialogues[did]:
                    fh.write(json.dumps(utt, ensure_ascii=False) + "\n")
                    count += 1
        counts[split_name] = count
        print(f"  {split_name}: {count} utterances from {len(ids)} dialogues")

    return counts


def prepare_for_training(
    split_jsonl: str | Path,
    output_path: str | Path,
    tokenizer_name: str = "google/muril-base-cased",
    max_length: int = 256,
    left_context: int = 2,
    right_context: int = 1,
) -> Dict[str, Any]:
    """
    Prepare a split for training by tokenizing and building contexts.

    CRITICAL: Each original utterance is one supervised target.
    - Emotion labels come directly from the utterance's emotion_label field.
    - Interaction-state and conduct-risk are IGNORE_INDEX (not in EmoInHindi).
    - Context is built from neighboring utterances in the same dialogue.
    - NO turn repair or text matching is used.

    Returns:
        Dict with num_examples, num_dialogues, label statistics.
    """
    import torch
    from transformers import AutoTokenizer
    from ..context_builder import _normalize_speaker, _role_token

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load utterances
    utterances = []
    with open(split_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            utterances.append(json.loads(line.strip()))

    # Group into dialogues
    dialogues: Dict[str, List[Dict]] = defaultdict(list)
    for utt in utterances:
        did = utt.get("dialogue_id", "default")
        dialogues[did].append(utt)

    all_input_ids = []
    all_attention_mask = []
    all_emotion_labels = []
    all_sentiment_labels = []
    all_intensity_labels = []
    all_is_labels = []
    all_is_masks = []
    all_cr_labels = []
    all_cr_masks = []

    emotion_counts = defaultdict(int)
    skipped = 0

    for did, utts in dialogues.items():
        utts.sort(key=lambda x: float(x.get("start", 0)))

        for target_idx, utt_data in enumerate(utts):
            # ── Build context from original utterances (NO repair_turns) ──
            context_parts = []

            # Left context (nearest first)
            left_start = max(0, target_idx - left_context)
            for j in range(left_start, target_idx):
                t = utts[j]
                role = _role_token(t.get("speaker", "unknown"))
                context_parts.append(f"{role} {t.get('text', '')}")

            # Target
            target_role = _role_token(utt_data.get("speaker", "unknown"), is_target=True)
            target_text = utt_data.get("text", "")
            context_parts.append(f"{target_role} {target_text}")

            # Right context
            right_end = min(len(utts), target_idx + right_context + 1)
            for j in range(target_idx + 1, right_end):
                t = utts[j]
                role = _role_token(t.get("speaker", "unknown"))
                context_parts.append(f"{role} {t.get('text', '')}")

            full_sequence = " ".join(context_parts)

            # Tokenize
            encoded = tokenizer(
                full_sequence,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            all_input_ids.append(encoded["input_ids"].squeeze(0))
            all_attention_mask.append(encoded["attention_mask"].squeeze(0))

            # ── Emotion labels (multi-label) — from this exact utterance ──
            emo_label = utt_data.get("emotion_label", ["neutral"])
            if isinstance(emo_label, str):
                emo_label = [e.strip() for e in emo_label.split(",")]
            emo_vec = torch.zeros(len(EMOINHINDI_EMOTIONS))
            for e in emo_label:
                if e in EMOINHINDI_EMOTIONS:
                    emo_vec[EMOINHINDI_EMOTIONS.index(e)] = 1.0
                    emotion_counts[e] += 1
                else:
                    print(f"  WARNING: unknown emotion '{e}' in {did}/{utt_data.get('_utterance_id', '?')}")
            all_emotion_labels.append(emo_vec)

            # ── Sentiment label — derived from primary emotion ──
            primary_emo = emo_label[0] if emo_label else "neutral"
            sentiment = utt_data.get("sentiment", EMOTION_TO_SENTIMENT.get(primary_emo, "neutral"))
            sent_idx = SENTIMENTS.index(sentiment) if sentiment in SENTIMENTS else 1
            all_sentiment_labels.append(torch.tensor(sent_idx))

            # ── Intensity — from this exact utterance ──
            intensity_data = utt_data.get("intensity", {})
            int_vec = torch.zeros(len(EMOINHINDI_EMOTIONS))
            if isinstance(intensity_data, dict):
                for e_name, val in intensity_data.items():
                    if e_name in EMOINHINDI_EMOTIONS:
                        int_vec[EMOINHINDI_EMOTIONS.index(e_name)] = float(val)
            all_intensity_labels.append(int_vec)

            # ── Interaction state — NOT in EmoInHindi, use IGNORE ──
            all_is_labels.append(torch.tensor(IGNORE_INDEX))
            all_is_masks.append(torch.tensor(False))

            # ── Conduct risk — NOT in EmoInHindi, use IGNORE ──
            all_cr_labels.append(torch.zeros(len(CONDUCT_RISKS)))
            all_cr_masks.append(torch.tensor(False))

    # Stack tensors
    data = {
        "input_ids": torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attention_mask),
        "emotion_labels": torch.stack(all_emotion_labels),
        "sentiment_labels": torch.stack(all_sentiment_labels),
        "intensity_labels": torch.stack(all_intensity_labels),
        "interaction_state_labels": torch.stack(all_is_labels),
        "interaction_state_mask": torch.stack(all_is_masks),
        "conduct_risk_labels": torch.stack(all_cr_labels),
        "conduct_risk_mask": torch.stack(all_cr_masks),
    }

    torch.save(data, output_path)

    stats = {
        "num_examples": len(all_input_ids),
        "num_dialogues": len(dialogues),
        "emotion_distribution": dict(emotion_counts),
        "skipped_utterances": skipped,
    }
    return stats


def _create_example_format(out_path: Path) -> None:
    """Create example data format documentation."""
    example = {
        "dialogue_id": "D001",
        "utterance_id": "U001",
        "speaker": "agent",
        "start": 0.0,
        "end": 2.5,
        "text": "Hello, how can I help you today?",
        "emotion_label": ["neutral"],
        "intensity": {"neutral": 0.9},
    }
    with open(out_path / "example_format.json", "w", encoding="utf-8") as fh:
        json.dump(example, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m svar.data.prepare_emoinhindi <data_dir> <output_dir>")
        sys.exit(1)

    counts = dialogue_split(sys.argv[1], sys.argv[2])
    print(f"Split counts: {counts}")
