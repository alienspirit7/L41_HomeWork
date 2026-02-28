"""
Nutrition lookup for single food items.

Primary: USDA FoodData Central API (free, no key needed for basic).
Fallback: local ``data/nutrition_db.json``.
"""

import json
import os
import urllib.request
import urllib.parse
from difflib import get_close_matches

_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "nutrition_db.json",
)
_local_db: dict | None = None

USDA_API = "https://api.nal.usda.gov/fdc/v1"
USDA_KEY = os.environ.get("USDA_API_KEY", "DEMO_KEY")


def _load_local_db() -> dict:
    """Load the local nutrition database (lazy, cached)."""
    global _local_db
    if _local_db is None:
        with open(_DB_PATH) as f:
            _local_db = json.load(f)
    return _local_db


def _usda_search(query: str) -> dict | None:
    """Search USDA FoodData Central for a food item.

    Returns per-100 g macros or None on failure.
    """
    try:
        params = urllib.parse.urlencode({
            "api_key": USDA_KEY,
            "query": query,
            "dataType": "Foundation,SR Legacy",
            "pageSize": 1,
        })
        url = f"{USDA_API}/foods/search?{params}"
        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())

        foods = data.get("foods", [])
        if not foods:
            return None

        nutrients = {
            n["nutrientName"]: n["value"]
            for n in foods[0].get("foodNutrients", [])
        }

        return {
            "carbs_g": round(nutrients.get("Carbohydrate, by difference", 0), 1),
            "protein_g": round(nutrients.get("Protein", 0), 1),
            "fat_g": round(nutrients.get("Total lipid (fat)", 0), 1),
            "calories": round(nutrients.get("Energy", 0), 1),
            "source": "usda_api",
            "usda_description": foods[0].get("description", query),
        }
    except Exception:
        return None


def _local_search(query: str) -> dict | None:
    """Fuzzy search the local nutrition database."""
    db = _load_local_db()

    # Exact match (case-insensitive)
    key = query.lower().strip()
    if key in db:
        result = dict(db[key])
        result["source"] = "local_db"
        return result

    # Fuzzy match
    matches = get_close_matches(key, db.keys(), n=1, cutoff=0.6)
    if matches:
        result = dict(db[matches[0]])
        result["source"] = "local_db"
        result["matched_name"] = matches[0]
        return result

    return None


def lookup_food(name: str) -> dict | None:
    """Look up nutrition per 100g for a food item.

    Tries USDA API first, falls back to local database.
    Returns dict with carbs_g, protein_g, fat_g, calories, source
    or None if not found.
    """
    # Try USDA API first
    result = _usda_search(name)
    if result:
        return result

    # Fallback to local database
    return _local_search(name)


def calculate_macros(name: str, weight_g: float) -> dict | None:
    """Look up a food and scale macros to the given weight.

    Returns dict with weight_g, carbs_g, protein_g, fat_g
    or None if food not found.
    """
    per100 = lookup_food(name)
    if per100 is None:
        return None

    scale = weight_g / 100.0
    return {
        "weight_g": round(weight_g, 1),
        "carbs_g": round(per100["carbs_g"] * scale, 1),
        "protein_g": round(per100["protein_g"] * scale, 1),
        "fat_g": round(per100["fat_g"] * scale, 1),
        "source": per100.get("source", "unknown"),
        "usda_description": per100.get(
            "usda_description",
            per100.get("matched_name", name),
        ),
    }


def list_foods() -> list[str]:
    """Return all food names in the local database."""
    db = _load_local_db()
    return sorted(db.keys())
