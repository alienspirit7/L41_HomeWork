"""
Multi-dish meal analysis blueprint.

POST /api/analyse_meal — accepts up to 5 dishes (1-3 images each),
returns per-dish predictions, meal totals, and combined bolus.
"""

import os
import io
import tempfile

from PIL import Image as PILImage
from flask import Blueprint, request, jsonify

# Register HEIC/HEIF support (iPhone images)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

from src.inference import predict_meal
from src.food_classifier import predict_food_name
from src.effective_carbs import bolus_recommendation_from_config

meal_bp = Blueprint("meal", __name__)

_cfg = None
_checkpoint = None

MAX_DISHES = 5
MAX_IMAGES_PER_DISH = 3


def init_meal_blueprint(cfg: dict, checkpoint: str):
    """Inject config and checkpoint path."""
    global _cfg, _checkpoint
    _cfg = cfg
    _checkpoint = checkpoint


def _save_upload(f) -> str:
    """Save an uploaded file as a clean PNG temp file.

    Reads raw bytes from the upload stream, opens with PIL
    (handles JPEG, PNG, HEIC, etc.), converts to RGB, and saves
    as PNG.  Returns the path to the temp PNG file.
    """
    f.stream.seek(0)
    raw = f.stream.read()

    if not raw:
        raise ValueError(f"Empty upload: {f.filename}")

    buf = io.BytesIO(raw)
    img = PILImage.open(buf).convert("RGB")

    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.save(path, format="PNG")
    img.close()
    return path


@meal_bp.route("/api/analyse_meal", methods=["POST"])
def analyse_meal():
    """Analyse a multi-dish meal."""
    dishes = []
    dish_idx = 0

    while dish_idx < MAX_DISHES:
        key = f"dish_{dish_idx}_images"
        files = request.files.getlist(key)
        if not files or not files[0].filename:
            break

        if len(files) > MAX_IMAGES_PER_DISH:
            return jsonify({
                "error": f"Dish {dish_idx + 1}: max "
                         f"{MAX_IMAGES_PER_DISH} images allowed",
            }), 400

        name_key = f"dish_{dish_idx}_name"
        dish_name = request.form.get(name_key, f"Dish {dish_idx + 1}")
        dishes.append({"files": files, "name": dish_name})
        dish_idx += 1

    if not dishes:
        return jsonify({"error": "No dishes provided"}), 400

    results = []
    all_temp_paths = []

    try:
        for dish in dishes:
            # Save each uploaded image as a clean PNG
            temp_paths = []
            for f in dish["files"]:
                path = _save_upload(f)
                temp_paths.append(path)
                all_temp_paths.append(path)

            # Predict macros
            ckpt = _checkpoint if os.path.exists(_checkpoint) else None
            prediction = predict_meal(
                temp_paths, _cfg, checkpoint=ckpt,
            )
            prediction.pop("bolus_recommendation", None)

            # Auto-predict dish name if user didn't provide one
            name = dish["name"]
            if not name or name.startswith("Dish "):
                try:
                    name = predict_food_name(temp_paths[0], _cfg)
                except Exception:
                    name = dish["name"]

            prediction["name"] = name
            prediction["num_images"] = len(temp_paths)
            results.append(prediction)

        # Meal totals
        total_weight = sum(d["weight_g"] for d in results)
        total_carbs = sum(d["carbs_g"] for d in results)
        total_protein = sum(d["protein_g"] for d in results)
        total_fat = sum(d["fat_g"] for d in results)

        bolus = bolus_recommendation_from_config(
            total_carbs, total_protein, total_fat, _cfg,
        )

        return jsonify({
            "dishes": results,
            "totals": {
                "weight_g": round(total_weight, 1),
                "carbs_g": round(total_carbs, 1),
                "protein_g": round(total_protein, 1),
                "fat_g": round(total_fat, 1),
                "num_dishes": len(results),
            },
            "bolus_recommendation": bolus,
        })

    finally:
        for p in all_temp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
