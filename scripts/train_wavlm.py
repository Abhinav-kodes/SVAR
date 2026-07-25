"""
Training script for WavLM audio emotion model.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml


class AudioEmotionDataset(Dataset):
    """Dataset for audio segments with emotion labels."""

    def __init__(self, data_dir: str, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.data_dir = Path(data_dir)

        # Load manifest
        manifest_path = self.data_dir / "manifest.jsonl"
        self.samples = []
        if manifest_path.exists():
            with open(manifest_path) as f:
                for line in f:
                    self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torchaudio
        sample = self.samples[idx]
        audio_path = self.data_dir / sample["audio_path"]

        waveform, sr = torchaudio.load(str(audio_path))
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return {
            "input_values": waveform.squeeze(0),
            "attention_mask": torch.ones(waveform.shape[-1], dtype=torch.long),
            "emotion_labels": torch.tensor(sample["emotion_label"], dtype=torch.float),
        }


def collate_audio(batch):
    """Pad audio to max length in batch."""
    max_len = max(b["input_values"].shape[0] for b in batch)
    input_ids = []
    masks = []
    labels = []

    for b in batch:
        pad_len = max_len - b["input_values"].shape[0]
        input_ids.append(
            torch.nn.functional.pad(b["input_values"], (0, pad_len))
        )
        masks.append(
            torch.nn.functional.pad(b["attention_mask"], (0, pad_len), value=0)
        )
        labels.append(b["emotion_labels"])

    return {
        "input_values": torch.stack(input_ids),
        "attention_mask": torch.stack(masks),
        "emotion_labels": torch.stack(labels),
    }


def train(config: Dict[str, Any]):
    """Main training loop for WavLM."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from svar.models.wavlm_emotion import WavLMEmotionModel
    model = WavLMEmotionModel(
        num_emotions=config.get("num_emotions", 16),
        freeze_base=config.get("freeze_base", True),
        dropout=config.get("dropout", 0.20),
    ).to(device)

    train_ds = AudioEmotionDataset(config["train_data_dir"])
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"],
        shuffle=True, collate_fn=collate_audio, num_workers=2,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
    )

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_values=batch["input_values"],
                attention_mask=batch["attention_mask"],
            )
            loss = nn.functional.binary_cross_entropy_with_logits(
                outputs["emotion_logits"], batch["emotion_labels"],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{config['epochs']} loss={total_loss/len(train_loader):.4f}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "wavlm_emotion.pt")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)
