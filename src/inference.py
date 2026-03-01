"""
Inference pipeline — load model, predict macros from images.

Supports 1–3 images per meal with configurable aggregation
(mean or max across views) and optional Test-Time Augmentation.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path

from src.model import FoodMacroModel
from src.transforms import get_eval_transforms, get_tta_transforms
from src.effective_carbs import (
    effective_carbs_from_config,
    bolus_recommendation_from_config,
)


# keyed by (checkpoint, device) → (model, target_mean, target_std)
_model_cache: dict = {}


def load_model(cfg: dict, checkpoint: str | None = None, device="cpu"):
    """Instantiate model and optionally load checkpoint.

    Caches the loaded model so repeated calls within the same process
    skip disk I/O and re-initialisation.

    Always re-injects target_mean/target_std from the cache into cfg
    so denormalisation works correctly on every call.

    Handles two checkpoint formats:
      - Dict with 'model_state_dict', 'target_mean', 'target_std'
      - Legacy plain state_dict
    """
    cache_key = (checkpoint, device)
    if cache_key in _model_cache:
        model, target_mean, target_std = _model_cache[cache_key]
        if target_mean is not None:
            cfg["target_mean"] = target_mean
        if target_std is not None:
            cfg["target_std"] = target_std
        return model

    model = FoodMacroModel(cfg)
    target_mean = None
    target_std = None
    if checkpoint and Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location=device,
                          weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            target_mean = ckpt.get("target_mean")
            target_std = ckpt.get("target_std")
            if target_mean is not None:
                cfg["target_mean"] = target_mean
            if target_std is not None:
                cfg["target_std"] = target_std
        else:
            model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    _model_cache[cache_key] = (model, target_mean, target_std)
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
def _predict_single(
    model, img: Image.Image, transform, device: str,
) -> np.ndarray:
    """Predict from a single image + transform."""
    tensor = transform(img).unsqueeze(0).to(device)
    return model(tensor).cpu().numpy()[0]


@torch.no_grad()
def _predict_tta(
    model, img: Image.Image, tta_transforms: list, device: str,
) -> np.ndarray:
    """Average predictions across TTA variants."""
    preds = []
    for t in tta_transforms:
        tensor = t(img).unsqueeze(0).to(device)
        pred = model(tensor).cpu().numpy()[0]
        preds.append(pred)
    return np.stack(preds).mean(axis=0)


@torch.no_grad()
def predict_meal(
    image_paths: list[str],
    cfg: dict,
    checkpoint: str | None = None,
    device: str = "cpu",
) -> dict:
    """Predict macros for a meal from 1–3 images.

    Returns dict with weight_g, carbs_g, protein_g, fat_g,
    effective_carbs_g, bolus_recommendation, and num_images.
    """
    model = load_model(cfg, checkpoint, device)
    use_tta = cfg.get("tta", False)

    if use_tta:
        tta_tfms = get_tta_transforms(cfg)
    else:
        eval_tfm = get_eval_transforms(cfg)

    preds_list = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        if use_tta:
            pred = _predict_tta(model, img, tta_tfms, device)
        else:
            pred = _predict_single(model, img, eval_tfm, device)
        preds_list.append(pred)

    preds = np.stack(preds_list)

    # Aggregate across views
    strategy = cfg.get("multi_image_strategy", "mean")
    if strategy == "mean":
        aggregated = preds.mean(axis=0)
    else:
        aggregated = preds.max(axis=0)

    # De-normalise
    raw_aggregated = aggregated.copy()
    aggregated = _denormalize(aggregated, cfg)

    import sys
    print(f"[INFERENCE] checkpoint={checkpoint}", flush=True, file=sys.stderr)
    print(f"[INFERENCE] target_mean={cfg.get('target_mean')}", flush=True, file=sys.stderr)
    print(f"[INFERENCE] raw_output={raw_aggregated}", flush=True, file=sys.stderr)
    print(f"[INFERENCE] denorm_output={aggregated}", flush=True, file=sys.stderr)

    weight, carbs, protein, fat = aggregated
    weight = max(0.0, float(weight))
    carbs = max(0.0, float(carbs))
    protein = max(0.0, float(protein))
    fat = max(0.0, float(fat))

    eff = effective_carbs_from_config(carbs, protein, fat, cfg)
    bolus = bolus_recommendation_from_config(carbs, protein, fat, cfg)

    return {
        "weight_g": round(weight, 1),
        "carbs_g": round(carbs, 1),
        "protein_g": round(protein, 1),
        "fat_g": round(fat, 1),
        "effective_carbs_g": round(eff, 1),
        "bolus_recommendation": bolus,
        "num_images": len(image_paths),
    }
