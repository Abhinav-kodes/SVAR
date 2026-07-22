import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List


LABEL_MAP = {
    "anger": "anger",
    "annoyed": "anger",
    "sad": "sadness",
    "guilty": "sadness",
    "fear": "fear",
    "apprehensive": "fear",
    "joy": "happiness",
    "grateful": "happiness",
    "impressed": "happiness",
    "compassion": "happiness",
    "disgusted": "disgust",
    "neutral": "neutral",
    "confident": "neutral",
    "anticipation": "neutral",
    "hopeful": "neutral",
    "surprised": "neutral",
}


def map_emotion_taxonomy(raw_emotion: str) -> str:
    """
    Maps fine-grained raw emotion string into standard 6-class dialogue emotion taxonomy.

    Args:
        raw_emotion: String containing raw emotion name.

    Returns:
        One of 6 core emotion classes: 'anger', 'sadness', 'fear', 'happiness', 'disgust', 'neutral'.
    """
    raw = str(raw_emotion).strip().lower()
    return LABEL_MAP.get(raw, "neutral")


def parse_primary_emotion(emotions_val: Any, intensities_val: Any) -> str:
    """
    Parses multi-label emotion strings and intensity values to select primary emotion label by highest intensity.

    Args:
        emotions_val: Comma-separated emotion names string or list.
        intensities_val: Comma-separated intensity values string or list.

    Returns:
        Mapped primary emotion string.
    """
    if pd.isna(emotions_val) or not str(emotions_val).strip():
        return "neutral"

    emotions = [e.strip().lower() for e in str(emotions_val).split(",") if e.strip()]
    if not emotions:
        return "neutral"

    if pd.isna(intensities_val) or not str(intensities_val).strip():
        return map_emotion_taxonomy(emotions[0])

    try:
        intensities = [int(i.strip()) for i in str(intensities_val).split(",") if i.strip()]
    except Exception:
        intensities = [1] * len(emotions)

    if len(intensities) < len(emotions):
        intensities.extend([1] * (len(emotions) - len(intensities)))

    max_idx = int(np.argmax(intensities[:len(emotions)]))
    primary_raw = emotions[max_idx]
    return map_emotion_taxonomy(primary_raw)


def build_context_windows(df: pd.DataFrame, window_size: int = 2) -> pd.DataFrame:
    """
    Constructs context windows by prepending previous dialogue utterances within the same dialogueId.

    Args:
        df: Input pandas DataFrame containing 'dialogueId' and 'utterance' (or 'text').
        window_size: Number of previous utterances to prepend (default: 2).

    Returns:
        DataFrame with added 'input_text' column.
    """
    df_copy = df.copy()

    text_col = "utterance" if "utterance" in df_copy.columns else ("text" if "text" in df_copy.columns else None)
    if text_col is None or "dialogueId" not in df_copy.columns:
        df_copy["input_text"] = df_copy[text_col] if text_col else ""
        return df_copy

    input_texts = []
    for dialogue_id, group in df_copy.groupby("dialogueId", sort=False):
        utterances = group[text_col].astype(str).tolist()
        for idx in range(len(utterances)):
            start_idx = max(0, idx - window_size)
            context = utterances[start_idx:idx]
            current = utterances[idx]
            if context:
                full_text = " [SEP] ".join(context) + " [SEP] " + current
            else:
                full_text = current
            input_texts.append(full_text)

    df_copy["input_text"] = input_texts
    return df_copy


def process_and_split_dataset(
    df: pd.DataFrame,
    output_dir: str,
    max_neutral: int = 10000,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Processes raw dataset, maps taxonomy, caps neutral samples, applies dialogue-level splitting, and exports CSVs.

    Args:
        df: Input raw DataFrame.
        output_dir: Output directory path to save train.csv, val.csv, test.csv.
        max_neutral: Maximum ceiling for neutral category samples.
        train_ratio: Proportion of dialogues for training set (default: 0.8).
        val_ratio: Proportion of dialogues for validation set (default: 0.1).
        seed: Random seed for reproducibility.

    Returns:
        Dict containing 'train', 'val', 'test' DataFrames.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_proc = df.copy()

    # 1. Parse primary emotion labels
    if "emotions" in df_proc.columns and "emoIntensity" in df_proc.columns:
        df_proc["mapped_label"] = [
            parse_primary_emotion(e, i) for e, i in zip(df_proc["emotions"], df_proc["emoIntensity"])
        ]
    elif "emotion" in df_proc.columns:
        df_proc["mapped_label"] = df_proc["emotion"].apply(map_emotion_taxonomy)
    elif "label" in df_proc.columns:
        df_proc["mapped_label"] = df_proc["label"].apply(map_emotion_taxonomy)
    else:
        df_proc["mapped_label"] = "neutral"

    # 2. Construct context windows
    df_proc = build_context_windows(df_proc, window_size=2)

    # 3. Cap neutral category samples
    neutral_mask = df_proc["mapped_label"] == "neutral"
    neutral_df = df_proc[neutral_mask]
    non_neutral_df = df_proc[~neutral_mask]

    if len(neutral_df) > max_neutral:
        neutral_df = neutral_df.sample(n=max_neutral, random_state=seed)

    processed_df = pd.concat([non_neutral_df, neutral_df]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # 4. Split by dialogueId to prevent dialogue leakage across splits
    if "dialogueId" in processed_df.columns:
        dialogue_ids = processed_df["dialogueId"].unique()
        np.random.seed(seed)
        np.random.shuffle(dialogue_ids)

        n_dialogues = len(dialogue_ids)
        n_train = int(n_dialogues * train_ratio)
        n_val = int(n_dialogues * val_ratio)

        train_ids = set(dialogue_ids[:n_train])
        val_ids = set(dialogue_ids[n_train:n_train + n_val])

        train_df = processed_df[processed_df["dialogueId"].isin(train_ids)].copy()
        val_df = processed_df[processed_df["dialogueId"].isin(val_ids)].copy()
        test_df = processed_df[~processed_df["dialogueId"].isin(train_ids | val_ids)].copy()
    else:
        np.random.seed(seed)
        shuffled = processed_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_df = shuffled.iloc[:n_train].copy()
        val_df = shuffled.iloc[n_train:n_train + n_val].copy()
        test_df = shuffled.iloc[n_train + n_val:].copy()

    # Save splits to CSV
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_path = os.path.join(repo_root, "data", "emotions", "LREC_EmoInHindi.csv")
    output_dir = os.path.join(repo_root, "sentiment", "data")

    if os.path.exists(dataset_path):
        print(f"Loading EmoInHindi dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} utterances across {df['dialogueId'].nunique() if 'dialogueId' in df.columns else 0} dialogues.")
        splits = process_and_split_dataset(df, output_dir=output_dir)
        print(f"Dataset successfully processed and exported to {output_dir}:")
        print(f"  - Train split: {len(splits['train'])} samples")
        print(f"  - Val split:   {len(splits['val'])} samples")
        print(f"  - Test split:  {len(splits['test'])} samples")
    else:
        print(f"Dataset file not found at: {dataset_path}")


if __name__ == "__main__":
    main()
