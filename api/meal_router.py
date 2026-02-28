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


@meal_bp.route("/api/foods", methods=["GET"])
def get_foods():
    """Return list of known food names for autocomplete."""
    return jsonify(list_foods())


@meal_bp.route("/api/analyse_meal", methods=["POST"])
def analyse_meal():
    """Analyse a multi-dish meal (single-item or composed)."""
    dishes = []
    dish_idx = 0

    while dish_idx < MAX_DISHES:
        mode_key = f"dish_{dish_idx}_mode"
        mode = request.form.get(mode_key, "composed")

        if mode == "single":
            # Single item: name + weight from form
            name = request.form.get(
                f"dish_{dish_idx}_name", "",
            ).strip()
            weight_str = request.form.get(
                f"dish_{dish_idx}_weight", "",
            )
            if not name:
                break
            dishes.append({
                "mode": "single", "name": name,
                "weight": weight_str,
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
        for dish in dishes:
            if dish["mode"] == "single":
                # ── Single item: USDA / local lookup ──
                try:
                    weight = float(dish["weight"])
                except (ValueError, TypeError):
                    return jsonify({
                        "error": f"Invalid weight for '{dish['name']}'",
                    }), 400

                macros = calculate_macros(dish["name"], weight)
                if macros is None:
                    return jsonify({
                        "error": f"Food not found: '{dish['name']}'. "
                                 "Try a different name.",
                    }), 404

                prediction = {
                    "weight_g": macros["weight_g"],
                    "carbs_g": macros["carbs_g"],
                    "protein_g": macros["protein_g"],
                    "fat_g": macros["fat_g"],
                    "name": dish["name"],
                    "num_images": 0,
                    "mode": "single",
                    "source": macros.get("source", ""),
                    "usda_description": macros.get(
                        "usda_description", dish["name"],
                    ),
                }
                results.append(prediction)

            else:
                # ── Composed dish: Nutrition5k model ──
                temp_paths = []
                for f in dish["files"]:
                    path = _save_upload(f)
                    temp_paths.append(path)
                    all_temp_paths.append(path)

                ckpt = (
                    _checkpoint
                    if os.path.exists(_checkpoint)
                    else None
                )
                prediction = predict_meal(
                    temp_paths, _cfg, checkpoint=ckpt,
                )
                prediction.pop("bolus_recommendation", None)
                prediction["name"] = dish["name"]
                prediction["num_images"] = len(temp_paths)
                prediction["mode"] = "composed"
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
