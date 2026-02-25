"""
Image transforms for training and evaluation.

Train: resize → color-jitter → random-flip → normalize.
Eval:  resize → normalize.
"""

from torchvision import transforms


def get_train_transforms(cfg: dict) -> transforms.Compose:
    """Augmented transforms for training."""
    size = cfg["image_size"]
    mean = cfg["image_mean"]
    std = cfg["image_std"]

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.05,
        ),
        transforms.RandomAffine(
            degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_eval_transforms(cfg: dict) -> transforms.Compose:
    """Deterministic transforms for validation / inference."""
    size = cfg["image_size"]
    mean = cfg["image_mean"]
    std = cfg["image_std"]

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
