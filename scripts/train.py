"""
Training script — CLI entry point.

Usage:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config
from src.model import FoodMacroModel
from src.data_utils import compute_target_stats, create_dataloaders
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train Food Macro Estimation model",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device: cpu, cuda, mps",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config loaded: backbone={cfg['backbone']}, "
          f"image_size={cfg['image_size']}")

    # Compute normalisation stats from training data
    print("Computing target statistics...")
    t_mean, t_std = compute_target_stats(cfg["data_csv"])
    cfg["target_mean"] = t_mean.tolist()
    cfg["target_std"] = t_std.tolist()
    print(f"  mean={t_mean}, std={t_std}")

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        cfg, target_mean=t_mean, target_std=t_std,
    )
    print(f"  Train: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")

    # Build model
    model = FoodMacroModel(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"Model: {total_params:,} params, "
          f"{trainable:,} trainable")

    # Train
    trainer = Trainer(model, cfg, device=args.device)
    trainer.fit(train_loader, val_loader)

    # Re-save best checkpoint with target stats included
    import torch
    ckpt_path = f"{cfg['checkpoint_dir']}/best.pt"
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu",
                                weights_only=True)
    else:
        # Fallback: save current model state if no best.pt yet
        state_dict = model.state_dict()
    torch.save({
        "model_state_dict": state_dict,
        "target_mean": t_mean.tolist(),
        "target_std": t_std.tolist(),
        "config": cfg,
    }, ckpt_path)

    print("\nDone. Best model saved to "
          f"{cfg['checkpoint_dir']}/best.pt")


if __name__ == "__main__":
    main()
