"""
Single-food classification using nateraw/food (ViT fine-tuned on Food-101).

Food-101 has 101 food-specific classes, vastly outperforming the ~21 food
classes available in ImageNet/ResNet-50.  This is a zero-training drop-in
replacement — we just swap the model and remap class labels.
"""

from PIL import Image
from transformers import pipeline

# ── Food-101 label → local nutrition DB key ──────────────────────────────────
# Full 101-class mapping; None means no matching entry in our nutrition DB.
FOOD101_TO_DB: dict[str, str | None] = {
    "apple_pie":              None,
    "baby_back_ribs":         None,
    "baklava":                None,
    "beef_carpaccio":         "beef steak",
    "beef_tartare":           "beef steak",
    "beet_salad":             "beet",
    "beignets":               None,
    "bibimbap":               "rice cooked",
    "bread_pudding":          "white bread",
    "breakfast_burrito":      None,
    "bruschetta":             "white bread",
    "caesar_salad":           "lettuce",
    "cannoli":                None,
    "caprese_salad":          "tomato",
    "carrot_cake":            "carrot",
    "ceviche":                "shrimp",
    "cheesecake":             "cream cheese",
    "cheese_plate":           "cheddar cheese",
    "chicken_curry":          "chicken breast",
    "chicken_quesadilla":     "chicken breast",
    "chicken_wings":          "chicken thigh",
    "chocolate_cake":         None,
    "chocolate_mousse":       None,
    "churros":                None,
    "clam_chowder":           None,
    "club_sandwich":          "white bread",
    "crab_cakes":             None,
    "creme_brulee":           None,
    "croque_madame":          "white bread",
    "cup_cakes":              None,
    "deviled_eggs":           "egg",
    "donuts":                 None,
    "dumplings":              None,
    "edamame":                "pea",
    "eggs_benedict":          "egg",
    "escargots":              None,
    "falafel":                "chickpea",
    "filet_mignon":           "beef steak",
    "fish_and_chips":         "tilapia",
    "foie_gras":              None,
    "french_fries":           "potato",
    "french_onion_soup":      "onion",
    "french_toast":           "white bread",
    "fried_calamari":         None,
    "fried_rice":             "rice cooked",
    "frozen_yogurt":          "yogurt",
    "garlic_bread":           "white bread",
    "gnocchi":                "potato",
    "greek_salad":            "tomato",
    "grilled_cheese_sandwich": "cheddar cheese",
    "grilled_salmon":         "salmon",
    "guacamole":              "avocado",
    "gyoza":                  None,
    "hamburger":              "ground beef",
    "hot_and_sour_soup":      None,
    "hot_dog":                "sausage",
    "huevos_rancheros":       "egg",
    "hummus":                 "hummus",
    "ice_cream":              None,
    "lasagna":                "pasta cooked",
    "lobster_bisque":         None,
    "lobster_roll_sandwich":  None,
    "macaroni_and_cheese":    "pasta cooked",
    "macarons":               None,
    "miso_soup":              "tofu",
    "mussels":                None,
    "nachos":                 None,
    "omelette":               "egg",
    "onion_rings":            "onion",
    "oysters":                None,
    "pad_thai":               "pasta cooked",
    "paella":                 "rice cooked",
    "pancakes":               None,
    "panna_cotta":            None,
    "peking_duck":            None,
    "pho":                    None,
    "pizza":                  None,
    "pork_chop":              "pork chop",
    "poutine":                "potato",
    "prime_rib":              "beef steak",
    "pulled_pork_sandwich":   "pork chop",
    "ramen":                  None,
    "ravioli":                "pasta cooked",
    "red_velvet_cake":        None,
    "risotto":                "rice cooked",
    "samosa":                 None,
    "sashimi":                "salmon",
    "scallops":               None,
    "seaweed_salad":          None,
    "shrimp_and_grits":       "shrimp",
    "spaghetti_bolognese":    "pasta cooked",
    "spaghetti_carbonara":    "pasta cooked",
    "spring_rolls":           None,
    "steak":                  "beef steak",
    "strawberry_shortcake":   "strawberry",
    "sushi":                  "rice cooked",
    "tacos":                  None,
    "takoyaki":               None,
    "tiramisu":               None,
    "tuna_tartare":           "tuna",
    "waffles":                None,
}

# ── Singleton pipeline ────────────────────────────────────────────────────────
_pipe = None


def _load_model():
    global _pipe
    if _pipe is not None:
        return

    print("[ViT-Food101] loading nateraw/food …")
    _pipe = pipeline(
        "image-classification",
        model="nateraw/food",
        top_k=None,       # return all 101 scores
    )
    mapped = sum(1 for v in FOOD101_TO_DB.values() if v is not None)
    print(f"[ViT-Food101] ready — {mapped} of 101 classes mapped to nutrition DB")


def classify_single_food(image_path: str, top_k: int = 5) -> list[dict]:
    """Classify a single food item using nateraw/food (ViT, Food-101).

    Returns a list of ``{name, confidence, food101_class}`` dicts
    for recognised classes that map to a local nutrition DB entry,
    sorted by descending confidence.  Returns [] if nothing maps.
    """
    try:
        _load_model()
    except Exception as e:
        print(f"[ViT-Food101] failed to load model: {e}")
        return []

    try:
        img = Image.open(image_path).convert("RGB")
        raw = _pipe(img)           # list of {label, score}, all 101 classes

        results = []
        for item in raw:
            label = item["label"]    # e.g. "grilled_salmon"
            db_name = FOOD101_TO_DB.get(label)
            if db_name is not None:
                results.append({
                    "name": db_name,
                    "confidence": round(float(item["score"]), 4),
                    "food101_class": label,
                })
                if len(results) >= top_k:
                    break

        if results:
            print(
                f"[ViT-Food101] top match: {results[0]['name']} "
                f"({results[0]['confidence']:.1%}, "
                f"Food-101: {results[0]['food101_class']})"
            )

        return results

    except Exception as e:
        print(f"[ViT-Food101] classification failed: {e}")
        return []


def get_all_predictions(image_path: str) -> list[dict]:
    """Return ALL Food-101 predictions (for debugging)."""
    try:
        _load_model()
    except Exception:
        return []

    try:
        img = Image.open(image_path).convert("RGB")
        raw = _pipe(img)
        return [
            {
                "food101_class": item["label"],
                "confidence": round(float(item["score"]), 4),
                "db_name": FOOD101_TO_DB.get(item["label"]),
            }
            for item in raw[:10]
        ]
    except Exception:
        return []
