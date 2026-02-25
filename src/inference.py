"""
Inference pipeline — load model, predict macros from images.

Supports 1–3 images per meal with configurable aggregation
(mean or max across views).
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path

from src.model import FoodMacroModel
from src.transforms import get_eval_transforms
from src.effective_carbs import effective_carbs_from_config


def load_model(cfg: dict, checkpoint: str | None = None, device="cpu"):
    """Instantiate model and optionally load checkpoint."""
    model = FoodMacroModel(cfg)
    if checkpoint and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=device,
                           weights_only=True)
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _denormalize(preds: np.ndarray, cfg: dict) -> np.ndarray:
    """Reverse z-score normalisation on predictions."""
    mean = cfg.get("target_mean")
    std = cfg.get("target_std")
    if mean is not None and std is not None:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        return preds * (std + 1e-8) + mean
    return preds


@torch.no_grad()
def predict_meal(
    image_paths: list[str],
    cfg: dict,
    checkpoint: str | None = None,
    device: str = "cpu",
) -> dict:
    """Predict macros for a meal from 1–3 images.

    Returns dict with weight_g, carbs_g, protein_g, fat_g,
    effective_carbs_g, and num_images.
    """
    model = load_model(cfg, checkpoint, device)
    transform = get_eval_transforms(cfg)

    preds_list = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        pred = model(tensor).cpu().numpy()[0]
        preds_list.append(pred)

    preds = np.stack(preds_list)

    # Aggregate across views
    strategy = cfg.get("multi_image_strategy", "mean")
    if strategy == "mean":
        aggregated = preds.mean(axis=0)
    else:
        aggregated = preds.max(axis=0)

    # De-normalise
    aggregated = _denormalize(aggregated, cfg)

    weight, carbs, protein, fat = aggregated
    weight = max(0.0, float(weight))
    carbs = max(0.0, float(carbs))
    protein = max(0.0, float(protein))
    fat = max(0.0, float(fat))

    eff = effective_carbs_from_config(carbs, protein, fat, cfg)

    return {
        "weight_g": round(weight, 1),
        "carbs_g": round(carbs, 1),
        "protein_g": round(protein, 1),
        "fat_g": round(fat, 1),
        "effective_carbs_g": round(eff, 1),
        "num_images": len(image_paths),
    }
