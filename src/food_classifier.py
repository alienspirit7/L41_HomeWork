"""
Food classification using CLIP zero-shot inference.

Classifies a food image against a combined label set of Food-101
categories and common single ingredients (from the local nutrition DB).
"""

import json
import os

import torch
import open_clip
from PIL import Image

# ── Food-101 class labels (spaces, not underscores) ──────────
FOOD_101_LABELS = [
    "apple pie", "baby back ribs", "baklava", "beef carpaccio",
    "beef tartare", "beet salad", "beignets", "bibimbap",
    "bread pudding", "breakfast burrito", "bruschetta",
    "caesar salad", "cannoli", "caprese salad", "carrot cake",
    "ceviche", "cheesecake", "cheese plate", "chicken curry",
    "chicken quesadilla", "chicken wings", "chocolate cake",
    "chocolate mousse", "churros", "clam chowder",
    "club sandwich", "crab cakes", "creme brulee",
    "croque madame", "cup cakes", "deviled eggs", "donuts",
    "dumplings", "edamame", "eggs benedict", "escargots",
    "falafel", "filet mignon", "fish and chips", "foie gras",
    "french fries", "french onion soup", "french toast",
    "fried calamari", "fried rice", "frozen yogurt",
    "garlic bread", "gnocchi", "greek salad",
    "grilled cheese sandwich", "grilled salmon", "guacamole",
    "gyoza", "hamburger", "hot and sour soup", "hot dog",
    "huevos rancheros", "hummus", "ice cream", "lasagna",
    "lobster bisque", "lobster roll sandwich",
    "macaroni and cheese", "macarons", "miso soup", "mussels",
    "nachos", "omelette", "onion rings", "oysters", "pad thai",
    "paella", "pancakes", "panna cotta", "peking duck", "pho",
    "pizza", "pork chop", "poutine", "prime rib",
    "pulled pork sandwich", "ramen", "ravioli",
    "red velvet cake", "risotto", "samosa", "sashimi",
    "scallops", "seaweed salad", "shrimp and grits",
    "spaghetti bolognese", "spaghetti carbonara",
    "spring rolls", "steak", "strawberry shortcake", "sushi",
    "tacos", "takoyaki", "tiramisu", "tuna tartare", "waffles",
]

# ── Singleton CLIP model ─────────────────────────────────────
_model = None
_preprocess = None
_tokenizer = None
_text_features = None
_labels = None
_logit_scale = None


def _load_model():
    """Load CLIP model and pre-compute text embeddings (lazy, cached)."""
    global _model, _preprocess, _tokenizer, _text_features, _labels, _logit_scale

    if _model is not None:
        return

    print("[CLIP] loading ViT-B-32 model …")
    _model, _, _preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k",
    )
    _tokenizer = open_clip.get_tokenizer("ViT-B-32")
    _model.eval()
    _logit_scale = _model.logit_scale.exp()

    # Build combined label set: Food-101 + local nutrition DB keys
    label_set = set(FOOD_101_LABELS)
    db_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "nutrition_db.json",
    )
    if os.path.exists(db_path):
        with open(db_path) as f:
            db = json.load(f)
        label_set.update(db.keys())

    _labels = sorted(label_set)

    # Pre-compute text embeddings for all labels
    prompts = [f"a photo of {name}, a type of food" for name in _labels]
    tokens = _tokenizer(prompts)
    with torch.no_grad():
        _text_features = _model.encode_text(tokens)
        _text_features /= _text_features.norm(dim=-1, keepdim=True)

    print(f"[CLIP] ready — {len(_labels)} food labels indexed")


# ── Ingredient-only CLIP embeddings (lazy) ───────────────────
_ingredient_labels = None
_ingredient_features = None

INGREDIENT_PROMPTS = [
    "a photo of a {}",
    "a {}, a single fresh ingredient",
    "a close-up photo of {}",
]


def _load_ingredient_embeddings():
    """Pre-compute CLIP embeddings for local DB labels only."""
    global _ingredient_labels, _ingredient_features

    if _ingredient_features is not None:
        return

    _load_model()

    db_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "nutrition_db.json",
    )
    with open(db_path) as f:
        db = json.load(f)
    _ingredient_labels = sorted(db.keys())

    # Prompt ensembling: average embeddings across templates
    all_features = []
    for template in INGREDIENT_PROMPTS:
        prompts = [template.format(name) for name in _ingredient_labels]
        tokens = _tokenizer(prompts)
        with torch.no_grad():
            feats = _model.encode_text(tokens)
            feats /= feats.norm(dim=-1, keepdim=True)
            all_features.append(feats)

    _ingredient_features = torch.stack(all_features).mean(dim=0)
    _ingredient_features /= _ingredient_features.norm(dim=-1, keepdim=True)

    print(f"[CLIP] ingredient embeddings ready — "
          f"{len(_ingredient_labels)} labels")


def classify_ingredient(image_path: str, top_k: int = 5) -> list[dict]:
    """Classify a single ingredient using CLIP with DB-only labels.

    Uses only local nutrition DB labels (~90 raw ingredients) instead
    of the full Food-101 + DB set.  This avoids probability dilution
    from dish categories (e.g. tomato vs. caprese salad).

    Uses prompt ensembling for better accuracy.
    """
    try:
        _load_ingredient_embeddings()
    except Exception as e:
        print(f"[CLIP] failed to load ingredient embeddings: {e}")
        return []

    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = _preprocess(img).unsqueeze(0)

        with torch.no_grad():
            image_features = _model.encode_image(img_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (
                _logit_scale * image_features @ _ingredient_features.T
            ).softmax(dim=-1)
            values, indices = similarity[0].topk(top_k)

        return [
            {"name": _ingredient_labels[idx],
             "confidence": round(float(val), 4)}
            for val, idx in zip(values, indices)
        ]
    except Exception as e:
        print(f"[CLIP] ingredient classification failed: {e}")
        return []


def classify_food(image_path: str, top_k: int = 3) -> list[dict]:
    """Classify food in an image using CLIP zero-shot inference.

    Returns a list of ``{name, confidence}`` dicts sorted by
    descending confidence.  Returns an empty list on failure.
    """
    try:
        _load_model()
    except Exception as e:
        print(f"[CLIP] failed to load model: {e}")
        return []

    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = _preprocess(img).unsqueeze(0)

        with torch.no_grad():
            image_features = _model.encode_image(img_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (_logit_scale * image_features @ _text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(top_k)

        return [
            {"name": _labels[idx], "confidence": round(float(val), 4)}
            for val, idx in zip(values, indices)
        ]
    except Exception as e:
        print(f"[CLIP] classification failed: {e}")
        return []
