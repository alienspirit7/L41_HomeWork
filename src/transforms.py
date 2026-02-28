"""
Image transforms for training and evaluation.

Train: resize → augment → normalize.
Eval:  resize → normalize.
TTA:   multiple augmented versions for inference averaging.
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
            brightness=0.3, contrast=0.3,
            saturation=0.3, hue=0.08,
        ),
        transforms.RandomAffine(
            degrees=15, translate=(0.08, 0.08),
            scale=(0.85, 1.15),
        ),
        transforms.RandomPerspective(
            distortion_scale=0.1, p=0.3,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
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


def get_tta_transforms(cfg: dict) -> list[transforms.Compose]:
    """5 TTA variants: original + 4 augmented views."""
    size = cfg["image_size"]
    mean = cfg["image_mean"]
    std = cfg["image_std"]

    base = [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    variants = [
        # Original
        transforms.Compose(base),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
        # Slight zoom in
        transforms.Compose([
            transforms.Resize((int(size * 1.1), int(size * 1.1))),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
        # Small rotation
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
        # Color shift
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
    ]
    return variants
