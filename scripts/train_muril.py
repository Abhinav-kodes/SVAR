"""
Training script for the SVAR emotion recognition system.

Supports multi-task training with weighted loss, early stopping,
and mixed precision.
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


class EmoInHindiDataset(Dataset):
    """Dataset loading pre-tokenized .pt files from prepare_emoinhindi."""

    def __init__(self, pt_path: str | Path):
        data = torch.load(pt_path, weights_only=False)
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.emotion_labels = data["emotion_labels"]
        self.sentiment_labels = data["sentiment_labels"]
        self.intensity_labels = data.get("intensity_labels")
        self.is_labels = data.get("interaction_state_labels")
        self.is_mask = data.get("interaction_state_mask")
        self.cr_labels = data.get("conduct_risk_labels")
        self.cr_mask = data.get("conduct_risk_mask")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "emotion_labels": self.emotion_labels[idx],
            "sentiment_labels": self.sentiment_labels[idx],
        }
        if self.intensity_labels is not None:
            item["intensity_labels"] = self.intensity_labels[idx]
        if self.is_labels is not None:
            item["interaction_state_labels"] = self.is_labels[idx]
        if self.is_mask is not None:
            item["interaction_state_mask"] = self.is_mask[idx]
        if self.cr_labels is not None:
            item["conduct_risk_labels"] = self.cr_labels[idx]
        if self.cr_mask is not None:
            item["conduct_risk_mask"] = self.cr_mask[idx]
        return item


def train_one_epoch(model, loader, optimizer, scheduler, device, loss_weights):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        # Emotion mask: all 1s for EmoInHindi (all labels present)
        emotion_mask = torch.ones_like(batch["emotion_labels"])

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            emotion_labels=batch["emotion_labels"],
            emotion_mask=emotion_mask,
            sentiment_labels=batch["sentiment_labels"],
            interaction_state_labels=batch.get("interaction_state_labels"),
            conduct_risk_labels=batch.get("conduct_risk_labels"),
            conduct_risk_mask=batch.get("conduct_risk_mask"),
            return_loss=True,
        )

        losses = outputs["losses"]
        # Weighted total
        weighted = sum(
            losses[k] * loss_weights.get(k, 1.0)
            for k in losses
        )

        optimizer.zero_grad()
        weighted.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += weighted.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, loss_weights):
    model.eval()
    total_loss = 0
    n_batches = 0
    emotion_correct = 0
    emotion_total = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        emotion_mask = torch.ones_like(batch["emotion_labels"])

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            emotion_labels=batch["emotion_labels"],
            emotion_mask=emotion_mask,
            sentiment_labels=batch["sentiment_labels"],
            conduct_risk_labels=batch.get("conduct_risk_labels"),
            conduct_risk_mask=batch.get("conduct_risk_mask"),
            return_loss=True,
        )

        losses = outputs["losses"]
        weighted = sum(
            losses[k] * loss_weights.get(k, 1.0)
            for k in losses
        )
        total_loss += weighted.item()
        n_batches += 1

        # Emotion accuracy (micro)
        probs = outputs["emotion_probs"]
        preds = (probs > 0.5).float()
        emotion_correct += (preds == batch["emotion_labels"]).sum().item()
        emotion_total += batch["emotion_labels"].numel()

    return {
        "loss": total_loss / max(n_batches, 1),
        "emotion_accuracy": emotion_correct / max(emotion_total, 1),
    }


def train(config: Dict[str, Any]):
    """Main training loop."""
    with open(config["data_config"]) as f:
        data_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    train_ds = EmoInHindiDataset(data_cfg["train_path"])
    val_ds = EmoInHindiDataset(data_cfg["val_path"])

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"],
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # Model
    from svar.models.contextual_muril import ContextualCallEmotionModel
    model = ContextualCallEmotionModel(
        model_name=config.get("model_name", "google/muril-base-cased"),
        num_emotions=config.get("num_emotions", 16),
        dropout=config.get("dropout", 0.20),
        freeze_base_layers=config.get("freeze_base_layers", 6),
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.01),
    )

    total_steps = len(train_loader) * config["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config["learning_rate"] * 0.1,
    )

    # Loss weights
    loss_weights = config.get("loss_weights", {
        "emotion": 1.0,
        "intensity": 0.3,
        "sentiment": 0.5,
        "interaction_state": 0.5,
        "conduct_risk": 0.3,
    })

    # Training loop
    best_val_loss = float("inf")
    patience = config.get("patience", 5)
    patience_counter = 0

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["epochs"]):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, loss_weights,
        )
        val_metrics = evaluate(model, val_loader, device, loss_weights)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1}/{config['epochs']} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_emotion_acc={val_metrics['emotion_accuracy']:.4f} "
            f"time={elapsed:.1f}s"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "config": config,
            }, output_dir / "best_model.pt")
            print(f"  Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Save final model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": config,
    }, output_dir / "final_model.pt")

    # Save training config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)
