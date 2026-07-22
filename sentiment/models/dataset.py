import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional


EMOTION_LABEL2ID = {
    "anger": 0,
    "sadness": 1,
    "fear": 2,
    "happiness": 3,
    "disgust": 4,
    "neutral": 5
}

EMOTION_ID2LABEL = {v: k for k, v in EMOTION_LABEL2ID.items()}

SENTIMENT_LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}


def map_emotion_to_sentiment(emotion_label: str) -> int:
    """Maps 6 emotion categories into 3 sentiment classes (positive, neutral, negative)."""
    emo = str(emotion_label).strip().lower()
    if emo in ["anger", "sadness", "fear", "disgust"]:
        return SENTIMENT_LABEL2ID["negative"]
    elif emo in ["happiness"]:
        return SENTIMENT_LABEL2ID["positive"]
    else:
        return SENTIMENT_LABEL2ID["neutral"]


class EmotionDataset(Dataset):
    """
    PyTorch Dataset class for tokenizing dialogue context & utterances for MuRIL multi-task training.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any = None,
        max_len: int = 128
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

        if "input_text" in self.df.columns:
            self.texts = self.df["input_text"].astype(str).tolist()
        elif "utterance" in self.df.columns:
            self.texts = self.df["utterance"].astype(str).tolist()
        else:
            self.texts = self.df.get("text", pd.Series([""] * len(self.df))).astype(str).tolist()

        raw_labels = self.df["mapped_label"].tolist() if "mapped_label" in self.df.columns else self.df.get("label", ["neutral"] * len(self.df)).tolist()
        self.emotion_ids = [EMOTION_LABEL2ID.get(str(lbl).strip().lower(), 5) for lbl in raw_labels]
        self.sentiment_ids = [map_emotion_to_sentiment(lbl) for lbl in raw_labels]

        if "emoIntensity" in self.df.columns:
            try:
                self.intensities = [float(str(i).split(",")[0]) if pd.notna(i) else 1.0 for i in self.df["emoIntensity"]]
            except Exception:
                self.intensities = [1.0] * len(self.df)
        else:
            self.intensities = [1.0] * len(self.df)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        if self.tokenizer is not None and callable(self.tokenizer):
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
        else:
            # Fallback mock encoding if tokenizer is None
            input_ids = torch.zeros(self.max_len, dtype=torch.long)
            attention_mask = torch.ones(self.max_len, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "emotion_label": torch.tensor(self.emotion_ids[idx], dtype=torch.long),
            "sentiment_label": torch.tensor(self.sentiment_ids[idx], dtype=torch.long),
            "intensity": torch.tensor(self.intensities[idx], dtype=torch.float32)
        }


def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Computes inverse class frequency weights tensor for emotion loss balancing."""
    raw_labels = df["mapped_label"].tolist() if "mapped_label" in df.columns else ["neutral"] * len(df)
    counts = np.zeros(6, dtype=np.float32)

    for lbl in raw_labels:
        idx = EMOTION_LABEL2ID.get(str(lbl).strip().lower(), 5)
        counts[idx] += 1.0

    total = np.sum(counts)
    counts = np.maximum(counts, 1.0)
    weights = total / (len(EMOTION_LABEL2ID) * counts)
    return torch.tensor(weights, dtype=torch.float32)
