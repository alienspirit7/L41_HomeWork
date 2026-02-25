"""
Trainer — training loop with two-phase schedule, validation,
early stopping, and checkpointing.
"""

import torch
from pathlib import Path
from src.losses import MultiTaskLoss
from src.metrics import mae, TARGET_NAMES
import numpy as np


class Trainer:
    """Two-phase trainer: frozen backbone → fine-tune top-N."""

    def __init__(self, model, cfg: dict, device: str = "cpu"):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.loss_fn = MultiTaskLoss(cfg)
        self.best_metric = float("inf")
        self.patience_counter = 0
        self.ckpt_dir = Path(cfg["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _build_optimizer(self, phase: str):
        """Build optimizer for the given phase."""
        lr = (self.cfg["lr_head"] if phase == "frozen"
              else self.cfg["lr_backbone"])
        params = filter(lambda p: p.requires_grad,
                        self.model.parameters())
        if self.cfg.get("optimizer", "adamw") == "adamw":
            return torch.optim.AdamW(
                params, lr=lr,
                weight_decay=self.cfg["weight_decay"],
            )
        return torch.optim.Adam(params, lr=lr)

    def train_epoch(self, loader, optimizer) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        for images, targets in loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            optimizer.zero_grad()
            preds = self.model(images)
            loss = self.loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def validate(self, loader) -> dict:
        """Run validation. Returns dict of per-target MAE."""
        self.model.eval()
        all_preds, all_targets = [], []
        for images, targets in loader:
            images = images.to(self.device)
            preds = self.model(images)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        mae_vals = mae(y_true, y_pred)
        return {
            f"val_{name}_mae": float(v)
            for name, v in zip(TARGET_NAMES, mae_vals)
        }

    def _check_early_stop(self, val_metrics: dict) -> bool:
        """Check early stopping. Returns True if should stop."""
        metric = self.cfg["early_stopping_metric"]
        current = val_metrics.get(metric, float("inf"))
        if current < self.best_metric:
            self.best_metric = current
            self.patience_counter = 0
            self._save_checkpoint("best.pt")
            return False
        self.patience_counter += 1
        return self.patience_counter >= self.cfg["early_stopping_patience"]

    def _save_checkpoint(self, name: str) -> None:
        path = self.ckpt_dir / name
        torch.save(self.model.state_dict(), path)

    def fit(self, train_loader, val_loader) -> None:
        """Full two-phase training loop."""
        # Phase 1: frozen backbone
        print("=== Phase 1: Training head (backbone frozen) ===")
        opt = self._build_optimizer("frozen")
        for epoch in range(self.cfg["epochs_frozen"]):
            loss = self.train_epoch(train_loader, opt)
            metrics = self.validate(val_loader)
            carb_mae = metrics.get("val_carbs_g_mae", 0)
            print(f"  Epoch {epoch+1}/{self.cfg['epochs_frozen']}  "
                  f"loss={loss:.4f}  carb_mae={carb_mae:.2f}")
            if self._check_early_stop(metrics):
                print("  Early stopping triggered.")
                break

        # Phase 2: unfreeze top-N layers
        print("=== Phase 2: Fine-tuning backbone top layers ===")
        self.model.unfreeze_backbone(self.cfg["unfreeze_top_n"])
        self.patience_counter = 0
        opt = self._build_optimizer("finetune")
        for epoch in range(self.cfg["epochs_finetune"]):
            loss = self.train_epoch(train_loader, opt)
            metrics = self.validate(val_loader)
            carb_mae = metrics.get("val_carbs_g_mae", 0)
            print(f"  Epoch {epoch+1}/{self.cfg['epochs_finetune']}  "
                  f"loss={loss:.4f}  carb_mae={carb_mae:.2f}")
            if self._check_early_stop(metrics):
                print("  Early stopping triggered.")
                break

        print("Training complete. Best checkpoint saved.")
