"""
Parse Nutrition5k metadata CSVs into unified processed.csv.

Reads dish_metadata_cafe1.csv and dish_metadata_cafe2.csv,
maps each dish to its overhead RGB image path, and outputs a
single CSV: data/nutrition5k/processed.csv

Columns: dish_id, image_path, weight_g, carbs_g, protein_g, fat_g
"""

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = Path("data/nutrition5k")
IMAGERY_DIR = DATA_DIR / "imagery" / "realsense_overhead"
OUTPUT_CSV = DATA_DIR / "processed.csv"


def parse_metadata_csv(csv_path: Path) -> list[dict]:
    """Parse a Nutrition5k dish metadata CSV.

    Format: dish_id, total_calories, total_mass, total_fat,
            total_carb, total_protein, num_ingrs, ...
    """
    dishes = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                dish = {
                    "dish_id": row[0].strip(),
                    "weight_g": float(row[2]),
                    "fat_g": float(row[3]),
                    "carbs_g": float(row[4]),
                    "protein_g": float(row[5]),
                }
                dishes.append(dish)
            except (ValueError, IndexError):
                continue
    return dishes


def find_image(dish_id: str) -> str | None:
    """Find overhead RGB image for a dish."""
    img_dir = IMAGERY_DIR / dish_id
    if not img_dir.exists():
        return None
    # Look for rgb.png
    rgb_path = img_dir / "rgb.png"
    if rgb_path.exists():
        return str(rgb_path.relative_to(DATA_DIR))
    # Fallback: any png/jpg in the directory
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        matches = list(img_dir.glob(ext))
        if matches:
            return str(matches[0].relative_to(DATA_DIR))
    return None


def main():
    all_dishes = []

    for csv_name in ["dish_metadata_cafe1.csv",
                     "dish_metadata_cafe2.csv"]:
        csv_path = DATA_DIR / "metadata" / csv_name
        if csv_path.exists():
            dishes = parse_metadata_csv(csv_path)
            all_dishes.extend(dishes)
            print(f"Parsed {len(dishes)} dishes from {csv_name}")

    # Match dishes to images
    matched = []
    for dish in all_dishes:
        img_path = find_image(dish["dish_id"])
        if img_path:
            dish["image_path"] = img_path
            matched.append(dish)

    print(f"Matched {len(matched)} / {len(all_dishes)} "
          f"dishes to images")

    # Data cleaning — filter outliers
    before = len(matched)
    matched = [
        d for d in matched
        if d["weight_g"] <= 800
        and d["carbs_g"] <= 200
        and (d["carbs_g"] + d["protein_g"] + d["fat_g"]) > 0
    ]
    print(f"After cleaning: {len(matched)} dishes "
          f"(removed {before - len(matched)} outliers)")

    # Write output CSV
    fieldnames = ["dish_id", "image_path", "weight_g",
                  "carbs_g", "protein_g", "fat_g"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matched)

    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
