"""
Multi-dish meal analysis blueprint.

POST /api/analyse_meal — accepts up to 5 dishes, each in one of two modes:
  - **single**:   image → CLIP classification → USDA lookup → weight estimate
  - **composed**: images → Nutrition5k model direct regression

Returns per-dish predictions, meal totals, and combined bolus recommendation.
"""

import os
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
from src.effective_carbs import bolus_recommendation_from_config
from src.food_classifier import classify_food
from src.nutrition_lookup import calculate_macros, list_foods, lookup_food

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
    """Save an uploaded file as a clean PNG temp file."""
    import subprocess

    print(f"[UPLOAD] filename={f.filename}, "
          f"content_type={f.content_type}, "
          f"stream_type={type(f.stream).__name__}")

    # Preserve original extension
    ext = ".jpg"
    if f.filename and "." in f.filename:
        ext = "." + f.filename.rsplit(".", 1)[-1].lower()

    # Step 1: save raw upload to disk
    fd, raw_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    f.save(raw_path)

    file_size = os.path.getsize(raw_path)
    with open(raw_path, "rb") as fh:
        header = fh.read(16)
    print(f"[STEP1] saved to {raw_path}, "
          f"size={file_size}, "
          f"magic={header[:8].hex()}")

    # Fix: strip stray bytes before image signature
    # (Flask multipart parser can leak CRLF into file data)
    SIGNATURES = [
        b"\xff\xd8\xff",       # JPEG
        b"\x89PNG",            # PNG
        b"\x00\x00\x00",       # HEIC/HEIF (ftyp box)
        b"RIFF",               # WebP
        b"GIF8",               # GIF
        b"BM",                 # BMP
    ]
    offset = 0
    for sig in SIGNATURES:
        pos = header.find(sig)
        if pos > 0:
            offset = pos
            break

    if offset > 0:
        print(f"[FIX] stripping {offset} leading bytes")
        with open(raw_path, "rb") as fh:
            data = fh.read()
        with open(raw_path, "wb") as fh:
            fh.write(data[offset:])
        file_size = os.path.getsize(raw_path)

    if file_size == 0:
        os.unlink(raw_path)
        raise ValueError(f"Empty upload: {f.filename}")

    # Step 2: try opening with PIL
    try:
        print(f"[STEP2] trying PIL.open({raw_path})...")
        img = PILImage.open(raw_path).convert("RGB")
        print(f"[STEP2] PIL OK: {img.size}")
    except Exception as e:
        print(f"[STEP2] PIL failed: {e}")
        # Step 3: fallback — macOS sips
        sips_out = raw_path + ".png"
        print(f"[STEP3] trying sips → {sips_out}")
        result = subprocess.run(
            ["sips", "-s", "format", "png", raw_path,
             "--out", sips_out],
            capture_output=True, text=True,
        )
        print(f"[STEP3] sips rc={result.returncode}, "
              f"stderr={result.stderr.strip()}")
        os.unlink(raw_path)
        if result.returncode != 0:
            raise ValueError(
                f"Cannot process '{f.filename}': "
                f"{result.stderr.strip()}"
            )
        raw_path = sips_out
        img = PILImage.open(raw_path).convert("RGB")
        print(f"[STEP3] PIL after sips OK: {img.size}")

    # Save as clean PNG
    fd2, path = tempfile.mkstemp(suffix=".png")
    os.close(fd2)
    img.save(path, format="PNG")
    img.close()
    print(f"[DONE] saved PNG: {path}, "
          f"size={os.path.getsize(path)}")

    # Clean up
    if os.path.exists(raw_path) and raw_path != path:
        try:
            os.unlink(raw_path)
        except OSError:
            pass

    return path


# ── Food list endpoint (autocomplete) ────────────────────────

@meal_bp.route("/api/foods", methods=["GET"])
def get_foods():
    """Return list of known food names for autocomplete."""
    return jsonify(list_foods())


# ── Main analysis endpoint ───────────────────────────────────

@meal_bp.route("/api/analyse_meal", methods=["POST"])
def analyse_meal():
    """Analyse a multi-dish meal.

    Each dish has a *mode*:

    **single** — single food item
      1. Image is classified with CLIP (Food-101 / local DB labels).
      2. Nutrition per 100 g is looked up from USDA or local DB.
      3. Weight is estimated by the Nutrition5k model.
      4. Final macros = nutrition_per_100g * estimated_weight / 100.
      Optional overrides: ``dish_N_name`` and ``dish_N_weight``
      skip the classifier / weight estimator respectively.

    **composed** — multi-ingredient dish (unchanged)
      Images are fed to the Nutrition5k model which directly predicts
      weight, carbs, protein, and fat.
    """
    dishes = []
    dish_idx = 0

    while dish_idx < MAX_DISHES:
        mode_key = f"dish_{dish_idx}_mode"
        mode = request.form.get(mode_key, "composed")

        if mode == "single":
            # Single item: image (for classify + weight) or name+weight
            key = f"dish_{dish_idx}_images"
            files = request.files.getlist(key)
            has_image = files and files[0].filename

            name = request.form.get(
                f"dish_{dish_idx}_name", "",
            ).strip()
            weight_str = request.form.get(
                f"dish_{dish_idx}_weight", "",
            ).strip()

            # Need at least an image or a name to proceed
            if not has_image and not name:
                break

            dishes.append({
                "mode": "single",
                "name": name,
                "weight": weight_str,
                "files": files if has_image else [],
            })

        else:
            # Composed dish: images from upload
            key = f"dish_{dish_idx}_images"
            files = request.files.getlist(key)
            if not files or not files[0].filename:
                break
            if len(files) > MAX_IMAGES_PER_DISH:
                return jsonify({
                    "error": f"Dish {dish_idx + 1}: max "
                             f"{MAX_IMAGES_PER_DISH} images",
                }), 400
            name_key = f"dish_{dish_idx}_name"
            dish_name = request.form.get(
                name_key, f"Dish {dish_idx + 1}",
            )
            dishes.append({
                "mode": "composed", "name": dish_name,
                "files": files,
            })

        dish_idx += 1

    if not dishes:
        return jsonify({"error": "No dishes provided"}), 400

    results = []
    all_temp_paths = []

    try:
        for dish_num, dish in enumerate(dishes, start=1):
            if dish["mode"] == "single":
                prediction = _analyse_single(dish, all_temp_paths, dish_num)
                if isinstance(prediction, tuple):
                    # Error response
                    return prediction
                results.append(prediction)
            else:
                prediction = _analyse_composed(dish, all_temp_paths)
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


# ── Per-dish analysis helpers ─────────────────────────────────

def _analyse_single(dish: dict, all_temp_paths: list, dish_num: int = 1) -> dict | tuple:
    """Single-item pipeline: classify → lookup → estimate weight → calc.

    Returns a prediction dict or a (jsonify(error), status) tuple.
    """
    food_name = dish["name"]
    weight_str = dish["weight"]
    files = dish["files"]

    temp_paths = []
    classification = []

    # Save uploaded image
    if files:
        path = _save_upload(files[0])
        temp_paths.append(path)
        all_temp_paths.append(path)

    # 1. Classify food name from image (if not manually provided)
    MIN_CLIP_CONFIDENCE = 0.10

    if not food_name and temp_paths:
        classification = classify_food(temp_paths[0], top_k=3)
        if classification and classification[0]["confidence"] >= MIN_CLIP_CONFIDENCE:
            food_name = classification[0]["name"]
            print(f"[SINGLE] CLIP classified: {food_name} "
                  f"({classification[0]['confidence']:.1%})")
        else:
            top = classification[0] if classification else None
            hint = (f" Best guess: '{top['name']}' "
                    f"({top['confidence']:.0%} confidence)."
                    if top else "")
            return (
                jsonify({"error": f"Dish {dish_num}: Could not confidently "
                                  "classify food from image (need ≥10% "
                                  f"confidence).{hint} Please enter the food "
                                  "name manually."}),
                422,
            )

    if not food_name:
        return (
            jsonify({"error": f"Dish {dish_num}: No food name or image provided."}),
            400,
        )

    # 2. Look up nutrition per 100 g
    macros_per100 = lookup_food(food_name)

    # If CLIP name didn't match, try the top-3 candidates
    if macros_per100 is None and classification:
        for candidate in classification[1:]:
            macros_per100 = lookup_food(candidate["name"])
            if macros_per100 is not None:
                food_name = candidate["name"]
                break

    if macros_per100 is None:
        return (
            jsonify({
                "error": f"Dish {dish_num}: Nutrition data not found for "
                         f"'{food_name}'. Try a different name.",
            }),
            404,
        )

    # 3. Estimate weight from image (if not manually provided)
    weight_g = None
    if weight_str:
        try:
            weight_g = float(weight_str)
        except (ValueError, TypeError):
            return (
                jsonify({"error": f"Invalid weight: '{weight_str}'"}),
                400,
            )

    if weight_g is None and temp_paths:
        # Use the Nutrition5k model to estimate weight
        ckpt = (
            _checkpoint if os.path.exists(_checkpoint) else None
        )
        model_pred = predict_meal(temp_paths, _cfg, checkpoint=ckpt)
        weight_g = model_pred["weight_g"]
        print(f"[SINGLE] model estimated weight: {weight_g}g")

    if weight_g is None:
        return (
            jsonify({"error": "No weight provided and no image for estimation."}),
            400,
        )

    weight_g = max(1.0, weight_g)

    # 4. Calculate final macros = per_100g * weight / 100
    scale = weight_g / 100.0
    prediction = {
        "weight_g": round(weight_g, 1),
        "carbs_g": round(macros_per100["carbs_g"] * scale, 1),
        "protein_g": round(macros_per100["protein_g"] * scale, 1),
        "fat_g": round(macros_per100["fat_g"] * scale, 1),
        "name": food_name,
        "num_images": len(temp_paths),
        "mode": "single",
        "source": macros_per100.get("source", ""),
        "usda_description": macros_per100.get(
            "usda_description", food_name,
        ),
    }

    # Include classification info for the frontend
    if classification:
        prediction["classification"] = [
            {"name": c["name"], "confidence": c["confidence"]}
            for c in classification[:3]
        ]

    return prediction


def _analyse_composed(dish: dict, all_temp_paths: list) -> dict:
    """Composed dish pipeline: Nutrition5k model direct regression."""
    temp_paths = []
    for f in dish["files"]:
        path = _save_upload(f)
        temp_paths.append(path)
        all_temp_paths.append(path)

    ckpt = (
        _checkpoint if os.path.exists(_checkpoint) else None
    )
    prediction = predict_meal(temp_paths, _cfg, checkpoint=ckpt)
    prediction.pop("bolus_recommendation", None)
    prediction["name"] = dish["name"]
    prediction["num_images"] = len(temp_paths)
    prediction["mode"] = "composed"
    return prediction
