"""
Backbone factory — create pretrained feature extractors via timm.

Supported backbones: efficientnet_b0 (default), resnet50,
mobilenetv2_100, mobilenetv3_large_100, efficientnet_b1.
"""

import timm

SUPPORTED = {
    "efficientnet_b0", "efficientnet_b1",
    "resnet50", "resnet34",
    "mobilenetv2_100", "mobilenetv3_large_100",
}


def get_backbone(
    name: str = "efficientnet_b0",
    pretrained: bool = True,
) -> tuple:
    """Return (backbone_model, feature_dim).

    The backbone outputs a flat feature vector (global-pooled).
    """
    if name not in SUPPORTED:
        raise ValueError(
            f"Unsupported backbone '{name}'. Choose from: {SUPPORTED}"
        )

    model = timm.create_model(
        name, pretrained=pretrained, num_classes=0,  # remove classifier
    )
    # Get feature dimension by inspecting the model
    feat_dim = model.num_features

    return model, feat_dim


def freeze_backbone(model) -> None:
    """Freeze all backbone parameters."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_top_n(model, n: int = 3) -> None:
    """Unfreeze the last n parameter groups.

    Works with most timm models by reversing the named-parameter
    list and unfreezing parameters until n unique layer-groups
    have been touched.
    """
    unfrozen_groups = set()
    for name, param in reversed(list(model.named_parameters())):
        group = name.rsplit(".", 1)[0]
        if group not in unfrozen_groups:
            unfrozen_groups.add(group)
        if len(unfrozen_groups) > n:
            break
        param.requires_grad = True
