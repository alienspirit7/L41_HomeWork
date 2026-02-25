"""
Configuration loader for Food Macro Estimation.

Loads a YAML config file, validates required keys, and provides
defaults. All hyperparameters flow from here.
"""

import yaml
from pathlib import Path

REQUIRED_KEYS = [
    "backbone", "image_size", "batch_size",
    "lr_head", "epochs_frozen",
    "loss_type", "lambda_weight", "lambda_carbs",
    "effective_carbs_alpha", "effective_carbs_beta",
]


def load_config(path: str = "configs/default.yaml") -> dict:
    """Load YAML config and validate required keys."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    # Apply defaults for optional keys
    defaults = {
        "pretrained": True,
        "freeze_backbone": True,
        "unfreeze_top_n": 3,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "num_workers": 4,
        "val_split": 0.15,
        "test_split": 0.10,
        "target_mean": None,
        "target_std": None,
        "epochs_finetune": 20,
        "lr_backbone": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "adamw",
        "lambda_protein": 1.0,
        "lambda_fat": 1.0,
        "early_stopping_patience": 7,
        "early_stopping_metric": "val_carbs_mae",
        "checkpoint_dir": "models",
        "save_best_only": True,
        "multi_image_strategy": "mean",
        "data_root": ".",
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    return cfg


if __name__ == "__main__":
    cfg = load_config()
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")
