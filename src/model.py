"""
FoodMacroModel — multi-task regression on food images.

Architecture:
  Backbone (EfficientNet-B0 default)
  → Global Average Pool (inside backbone via num_classes=0)
  → Dense 512 → ReLU → Dropout 0.3
  → Dense 128 → ReLU
  → Dense 4 (linear) → [weight, carbs, protein, fat]
"""

import torch
import torch.nn as nn
from src.backbone import get_backbone, freeze_backbone, unfreeze_top_n


class FoodMacroModel(nn.Module):
    """Multi-task regression model for food macros."""

    def __init__(self, cfg: dict):
        super().__init__()
        name = cfg.get("backbone", "efficientnet_b0")
        pretrained = cfg.get("pretrained", True)

        self.backbone, feat_dim = get_backbone(name, pretrained)

        if cfg.get("freeze_backbone", True):
            freeze_backbone(self.backbone)

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),  # [weight, carbs, protein, fat]
        )

        # Initialise head weights
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: images → [weight, carbs, protein, fat]."""
        features = self.backbone(x)          # (B, feat_dim)
        return self.head(features)           # (B, 4)

    def unfreeze_backbone(self, n: int = 3) -> None:
        """Unfreeze top-N backbone layer groups for fine-tuning."""
        unfreeze_top_n(self.backbone, n)
