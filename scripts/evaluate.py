"""
Evaluate trained model on the held-out test set.

Loads the same deterministic data splits used during training,
runs inference on the test split, and prints MAE / RMSE / MAPE / Bias.

Usage:
    python scripts/evaluate.py \
        --config configs/default.yaml \
        --checkpoint models/best.pt \
        --device cpu
"""

import argparse
import sys
import os

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config
from src.model import FoodMacroModel
from src.data_utils import create_dataloaders
from src.metrics import compute_all_metrics, format_metrics, TARGET_NAMES


def load_checkpoint(path: str, device: str = "cpu"):
    """Load checkpoint and return model state + target stats."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return (
            ckpt["model_state_dict"],
            np.array(ckpt.get("target_mean", [0, 0, 0, 0]),
                     dtype=np.float32),
            np.array(ckpt.get("target_std", [1, 1, 1, 1]),
                     dtype=np.float32),
        )
    # Legacy plain state dict — no stats available
    return ckpt, None, None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on the test set",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
    )
    parser.add_argument(
        "--checkpoint", default="models/best.pt",
    )
    parser.add_argument(
        "--device", default="cpu",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Load checkpoint
    state_dict, t_mean, t_std = load_checkpoint(
        args.checkpoint, args.device,
    )

    if t_mean is None:
        print("WARNING: checkpoint has no target stats. "
              "De-normalisation will be skipped.")
        t_mean = np.zeros(4, dtype=np.float32)
        t_std = np.ones(4, dtype=np.float32)

    cfg["target_mean"] = t_mean.tolist()
    cfg["target_std"] = t_std.tolist()

    # Build model
    model = FoodMacroModel(cfg)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    print(f"Model loaded from {args.checkpoint}")

    # Create data loaders (deterministic splits — same as training)
    _, _, test_loader = create_dataloaders(
        cfg, target_mean=t_mean, target_std=t_std,
    )
    print(f"Test set: {len(test_loader.dataset)} samples\n")

    # Run inference on test set
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(args.device)
            preds = model(images)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    # De-normalise back to grams for meaningful metrics
    y_pred_grams = y_pred * (t_std + 1e-8) + t_mean
    y_true_grams = y_true * (t_std + 1e-8) + t_mean

    # Clamp predictions to >= 0
    y_pred_grams = np.maximum(y_pred_grams, 0)

    # Compute and display metrics
    metrics = compute_all_metrics(y_true_grams, y_pred_grams)
    print("=" * 56)
    print("  TEST SET EVALUATION RESULTS")
    print("=" * 56)
    print(format_metrics(metrics))
    print("=" * 56)

    # Print per-sample predictions for small test sets
    if len(y_pred_grams) <= 20:
        print(f"\n{'Sample':<8} ", end="")
        for name in TARGET_NAMES:
            short = name.replace("_g", "")
            print(f"{'True_'+short:>12} {'Pred_'+short:>12} ", end="")
        print()
        print("-" * 105)
        for i in range(len(y_pred_grams)):
            print(f"  {i+1:<6} ", end="")
            for j in range(4):
                print(f"{y_true_grams[i,j]:>12.1f} "
                      f"{y_pred_grams[i,j]:>12.1f} ", end="")
            print()


if __name__ == "__main__":
    main()
