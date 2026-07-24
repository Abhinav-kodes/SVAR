import os
import csv
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from typing import Dict, Any, Optional
from sklearn.metrics import f1_score, classification_report

from sentiment.models.muril_model import MultiTaskMuRIL
from sentiment.models.dataset import (
    EmotionDataset,
    compute_class_weights,
    EMOTION_ID2LABEL,
    SENTIMENT_LABEL2ID,
)

SENTIMENT_ID2LABEL = {v: k for k, v in SENTIMENT_LABEL2ID.items()}


def build_optimizer(
    named_parameters,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {
            "params": [p for n, p in named_parameters
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in named_parameters
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(params, lr=lr)


def build_loaders(
    train_csv: str,
    val_csv: str,
    tokenizer,
    max_len: int = 128,
    batch_size: int = 4,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    import pandas as pd

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    train_ds = EmotionDataset(train_df, tokenizer=tokenizer, max_len=max_len)
    val_ds = EmotionDataset(val_df, tokenizer=tokenizer, max_len=max_len)

    class_weights = compute_class_weights(train_df)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, class_weights


class MultiTaskLoss(nn.Module):
    """Weighted sum of emotion CE + sentiment CE + intensity MSE losses."""

    def __init__(self, emotion_weights: torch.Tensor, w_emotion: float = 1.0, w_sentiment: float = 0.5, w_intensity: float = 0.3):
        super().__init__()
        self.w_emotion = w_emotion
        self.w_sentiment = w_sentiment
        self.w_intensity = w_intensity
        self.emotion_criterion = nn.CrossEntropyLoss(weight=emotion_weights)
        self.sentiment_criterion = nn.CrossEntropyLoss()
        self.intensity_criterion = nn.MSELoss()

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        emotion_loss = self.emotion_criterion(outputs["emotion_logits"], batch["emotion_label"])
        sentiment_loss = self.sentiment_criterion(outputs["sentiment_logits"], batch["sentiment_label"])
        intensity_loss = self.intensity_criterion(outputs["intensity_pred"].squeeze(-1), batch["intensity"])
        total = self.w_emotion * emotion_loss + self.w_sentiment * sentiment_loss + self.w_intensity * intensity_loss
        return total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_emo_preds = []
    all_emo_labels = []
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = loss_fn(outputs, batch)
        total_loss += loss.item()
        n_batches += 1

        preds = outputs["emotion_logits"].argmax(dim=-1).cpu().tolist()
        labels = batch["emotion_label"].cpu().tolist()
        all_emo_preds.extend(preds)
        all_emo_labels.extend(labels)

    macro_f1 = f1_score(all_emo_labels, all_emo_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_emo_labels, all_emo_preds, average="micro", zero_division=0)
    avg_loss = total_loss / max(n_batches, 1)

    return {
        "loss": avg_loss,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "emotion_preds": all_emo_preds,
        "emotion_labels": all_emo_labels,
    }


def train(
    train_csv: str,
    val_csv: str,
    output_dir: str,
    model_name: Optional[str] = None,
    max_len: int = 64,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    num_epochs: int = 5,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    early_stopping_patience: int = 2,
    grad_clip: float = 1.0,
    freeze_encoder: bool = False,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Tokenizer
    from transformers import AutoTokenizer
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    local_muril = os.path.join(repo_root, "muril-base")
    tokenizer_src = local_muril if os.path.exists(local_muril) else (model_name or "google/muril-base-cased")
    print(f"Loading tokenizer from: {tokenizer_src}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)

    # Data
    train_loader, val_loader, class_weights = build_loaders(
        train_csv, val_csv, tokenizer, max_len=max_len, batch_size=batch_size,
    )
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # Model
    model = MultiTaskMuRIL(model_name=model_name if model_name else None).to(device)

    # Freeze encoder to save VRAM on small GPUs (only heads are trained)
    if freeze_encoder and hasattr(model, "encoder") and model.encoder is not None:
        for param in model.encoder.parameters():
            param.requires_grad = False
        model.encoder.eval()
        print("Encoder FROZEN — only training task heads.")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Enable gradient checkpointing if encoder exists (RTX 2050 VRAM constraint)
    if not freeze_encoder and hasattr(model, "encoder") and model.encoder is not None and hasattr(model.encoder, "gradient_checkpointing_enable"):
        try:
            model.encoder.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled on encoder.")
        except Exception:
            pass

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} total params, {trainable:,} trainable")

    class_weights = class_weights.to(device)
    loss_fn = MultiTaskLoss(emotion_weights=class_weights).to(device)
    optimizer = build_optimizer(
        [(n, p) for n, p in model.named_parameters() if p.requires_grad], lr=lr,
    )

    # Cosine schedule with linear warmup (account for grad accumulation)
    effective_steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps = effective_steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    amp_device = device if device == "cuda" else "cpu"
    scaler = GradScaler(amp_device, enabled=(device == "cuda"))

    # Training loop
    best_val_f1 = 0.0
    patience_counter = 0
    history = []

    print(f"\nStarting training: {num_epochs} epochs, {len(train_loader)} batches/epoch, grad_accum={grad_accum_steps}, warmup={warmup_steps}")
    print(f"{'Epoch':<6} {'Step':<6} {'Train Loss':<12} {'Val Loss':<12} {'Val Macro-F1':<13} {'LR':<12}")
    print("-" * 65)

    global_step = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for micro_step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(amp_device, enabled=(device == "cuda"), dtype=torch.float16):
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = loss_fn(outputs, batch) / grad_accum_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * grad_accum_steps

            if micro_step % grad_accum_steps == 0 or micro_step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        train_avg = epoch_loss / max(micro_step, 1)
        elapsed = time.time() - t0

        # Validation
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        current_lr = scheduler.get_last_lr()[0]

        row = {
            "epoch": epoch,
            "train_loss": round(train_avg, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_macro_f1": round(val_metrics["macro_f1"], 4),
            "val_micro_f1": round(val_metrics["micro_f1"], 4),
            "lr": current_lr,
            "time_s": round(elapsed, 1),
        }
        history.append(row)

        print(f"{epoch:<6} {global_step:<6} {train_avg:<12.4f} {val_metrics['loss']:<12.4f} {val_metrics['macro_f1']:<13.4f} {current_lr:<12.2e}  ({elapsed:.0f}s)")

        # Early stopping + checkpointing
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            ckpt_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_macro_f1": best_val_f1,
                "val_metrics": val_metrics,
            }, ckpt_path)
            print(f"  → Best model saved (macro-F1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
                break

    # Save training history CSV
    history_path = os.path.join(output_dir, "training_history.csv")
    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    # Final classification report on val set (using best checkpoint)
    best_ckpt = torch.load(os.path.join(output_dir, "best_model.pt"), map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])
    final = evaluate(model, val_loader, loss_fn, device)
    target_names = [EMOTION_ID2LABEL[i] for i in range(6)]
    report = classification_report(final["emotion_labels"], final["emotion_preds"], target_names=target_names, zero_division=0)
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n{'='*65}")
    print(f"Training complete. Best val macro-F1: {best_val_f1:.4f}")
    print(f"Outputs saved to: {output_dir}")
    print(f"\nClassification Report (best checkpoint):\n{report}")

    return {
        "best_val_f1": best_val_f1,
        "history": history,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MuRIL multi-task emotion classifier")
    parser.add_argument("--train_csv", default=os.path.join(os.path.dirname(__file__), "..", "data", "train.csv"))
    parser.add_argument("--val_csv", default=os.path.join(os.path.dirname(__file__), "..", "data", "val.csv"))
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "..", "models", "checkpoints"))
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_epochs=args.epochs,
        lr=args.lr,
        freeze_encoder=args.freeze_encoder,
        device=args.device,
    )
