"""
Lightweight food name classifier using pretrained ImageNet weights.

Uses the same EfficientNet backbone to predict a human-readable
food name from the 1000 ImageNet classes. No extra training needed.
"""

import torch
import timm
from PIL import Image
from src.transforms import get_eval_transforms

# ImageNet food-related classes (index → friendly name).
# Curated subset of the 1000 ImageNet categories.
FOOD_LABELS = {
    567: "Fried rice", 924: "Guacamole", 927: "Cup / Mug",
    928: "Cup / Mug", 929: "Plate",
    923: "Hot dog", 925: "Hot pot", 926: "Burrito",
    930: "Espresso", 931: "Pizza slice",
    932: "Ice cream", 933: "Ice lolly",
    934: "French loaf", 935: "Bagel",
    936: "Pretzel", 937: "Cheeseburger",
    938: "Hot dog", 939: "Mashed potato",
    940: "Broccoli", 941: "Cauliflower",
    942: "Courgette", 943: "Spaghetti squash",
    944: "Butternut squash", 945: "Cucumber",
    946: "Artichoke", 947: "Pepper",
    948: "Cardoon", 949: "Mushroom",
    950: "Granny Smith apple", 951: "Strawberry",
    952: "Orange", 953: "Lemon",
    954: "Fig", 955: "Pineapple",
    956: "Banana", 957: "Jackfruit",
    958: "Custard apple", 959: "Pomegranate",
    960: "Hay", 961: "Carbonara",
    962: "Chocolate sauce", 963: "Dough",
    964: "Meat loaf", 965: "Cheese / Pizza",
    966: "Waffle", 967: "Trifle",
    968: "Ice cream", 969: "Ice lolly / Popsicle",
    541: "Drumstick / Chicken", 542: "Dumbbell",
    553: "Frying pan / Stir fry", 559: "Grocery bag",
    566: "French horn", 568: "Bread basket",
}

# Fallback: full ImageNet label list (loaded on demand)
_imagenet_labels = None


def _load_imagenet_labels() -> list[str]:
    """Load all 1000 ImageNet class labels."""
    global _imagenet_labels
    if _imagenet_labels is not None:
        return _imagenet_labels

    try:
        from timm.data import ImageNetInfo
        info = ImageNetInfo()
        _imagenet_labels = [
            info.label_names[i] for i in range(1000)
        ]
    except Exception:
        # Fallback: use timm's built-in
        _imagenet_labels = [f"class_{i}" for i in range(1000)]

    return _imagenet_labels


def _clean_label(raw: str) -> str:
    """Clean up an ImageNet label to be user-friendly."""
    # Take first part before comma (e.g. "pizza, pizza pie" → "pizza")
    name = raw.split(",")[0].strip()
    # Capitalise
    return name.replace("_", " ").title()


_classifier_model = None
_classifier_cfg = None


@torch.no_grad()
def predict_food_name(
    image_path: str,
    cfg: dict,
    device: str = "cpu",
    top_k: int = 1,
) -> str:
    """Predict a human-readable food name from an image.

    Returns the most likely food name as a string.
    """
    global _classifier_model, _classifier_cfg

    backbone = cfg.get("backbone", "efficientnet_b2")

    # Lazy-load classifier (with head intact)
    if _classifier_model is None or _classifier_cfg != backbone:
        _classifier_model = timm.create_model(
            backbone, pretrained=True, num_classes=1000,
        )
        _classifier_model.to(device)
        _classifier_model.eval()
        _classifier_cfg = backbone

    transform = get_eval_transforms(cfg)
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    logits = _classifier_model(tensor)
    probs = torch.softmax(logits, dim=1)
    top_prob, top_idx = probs.topk(top_k, dim=1)

    idx = top_idx[0][0].item()

    # Check curated food labels first
    if idx in FOOD_LABELS:
        return FOOD_LABELS[idx]

    # Fallback to ImageNet labels
    labels = _load_imagenet_labels()
    return _clean_label(labels[idx])
