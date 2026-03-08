"""
Single-food classification using pre-trained ResNet-50 (ImageNet-1K).

ImageNet contains ~30 direct food classes (fruits, vegetables, etc.)
which are mapped to local nutrition DB keys.  Much more reliable than
CLIP for raw single ingredients because the model was explicitly
trained on these classes.
"""

import json
import os

import torch
from torchvision import models, transforms
from PIL import Image

# ── ImageNet class index → local nutrition DB name ──────────────
# Only classes that map to a single-ingredient item in our DB.
IMAGENET_TO_DB = {
    924: "guacamole",
    928: "ice cream",
    933: "hamburger",          # cheeseburger → close
    935: "potato",             # mashed potato
    936: "cabbage",            # head cabbage
    937: "broccoli",
    938: "cauliflower",
    939: "zucchini",           # zucchini / courgette
    943: "cucumber",
    945: "green pepper",       # bell pepper → green pepper
    947: "mushroom",
    948: "apple",              # Granny Smith
    949: "strawberry",
    950: "orange",
    951: "lemon",
    952: "fig",
    953: "pineapple",
    954: "banana",
    957: "pomegranate",
    959: "pasta cooked",       # carbonara → pasta
    987: "corn",
}

# ── Singleton model ─────────────────────────────────────────────
_model = None
_preprocess = None
_categories = None


def _load_model():
    """Load pre-trained ResNet-50 with ImageNet-1K weights (lazy)."""
    global _model, _preprocess, _categories

    if _model is not None:
        return

    print("[ResNet] loading ResNet-50 (ImageNet-1K) …")
    weights = models.ResNet50_Weights.DEFAULT
    _model = models.resnet50(weights=weights)
    _model.eval()

    _preprocess = weights.transforms()
    _categories = weights.meta["categories"]

    print(f"[ResNet] ready — {len(IMAGENET_TO_DB)} food classes mapped")


def classify_single_food(
    image_path: str,
    top_k: int = 5,
) -> list[dict]:
    """Classify a single food item using pre-trained ResNet-50.

    Returns a list of ``{name, confidence, imagenet_class}`` dicts
    for recognised food classes, sorted by descending confidence.
    Only returns matches that map to local nutrition DB items.
    Returns an empty list if no food class is recognised.
    """
    try:
        _load_model()
    except Exception as e:
        print(f"[ResNet] failed to load model: {e}")
        return []

    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = _preprocess(img).unsqueeze(0)

        with torch.no_grad():
            logits = _model(img_tensor)
            probs = torch.softmax(logits, dim=-1)

        # Get top-k across ALL 1000 classes first
        values, indices = probs[0].topk(min(top_k * 3, 50))

        results = []
        for val, idx in zip(values, indices):
            idx_int = int(idx)
            db_name = IMAGENET_TO_DB.get(idx_int)
            if db_name is not None:
                results.append({
                    "name": db_name,
                    "confidence": round(float(val), 4),
                    "imagenet_class": _categories[idx_int],
                })
                if len(results) >= top_k:
                    break

        if results:
            print(f"[ResNet] top match: {results[0]['name']} "
                  f"({results[0]['confidence']:.1%}, "
                  f"ImageNet: {results[0]['imagenet_class']})")

        return results

    except Exception as e:
        print(f"[ResNet] classification failed: {e}")
        return []


def get_all_predictions(image_path: str) -> list[dict]:
    """Return ALL ImageNet predictions (for debugging)."""
    try:
        _load_model()
    except Exception:
        return []

    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = _preprocess(img).unsqueeze(0)

        with torch.no_grad():
            logits = _model(img_tensor)
            probs = torch.softmax(logits, dim=-1)

        values, indices = probs[0].topk(10)
        return [
            {
                "index": int(idx),
                "class": _categories[int(idx)],
                "confidence": round(float(val), 4),
                "db_name": IMAGENET_TO_DB.get(int(idx)),
            }
            for val, idx in zip(values, indices)
        ]
    except Exception:
        return []
