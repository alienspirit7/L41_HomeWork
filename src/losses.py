"""
Multi-task loss for food macro regression.

Combines per-target Smooth-L1 (Huber) or MSE losses with
configurable lambda weights.
"""

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """Weighted sum of per-target regression losses.

    Loss = λ_w·L_weight + λ_c·L_carbs + λ_p·L_protein + λ_f·L_fat
    """

    def __init__(self, cfg: dict):
        super().__init__()
        loss_type = cfg.get("loss_type", "smooth_l1")

        if loss_type == "smooth_l1":
            self.criterion = nn.SmoothL1Loss(reduction="mean")
        elif loss_type == "mse":
            self.criterion = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        self.lambdas = torch.tensor([
            cfg.get("lambda_weight", 1.0),
            cfg.get("lambda_carbs", 2.0),
            cfg.get("lambda_protein", 1.0),
            cfg.get("lambda_fat", 1.0),
        ], dtype=torch.float32)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted multi-task loss.

        Args:
            pred:   (B, 4) predicted [weight, carbs, protein, fat]
            target: (B, 4) ground truth

        Returns:
            Scalar loss tensor.
        """
        lambdas = self.lambdas.to(pred.device)
        # Per-task losses
        losses = torch.stack([
            self.criterion(pred[:, i], target[:, i])
            for i in range(4)
        ])
        return (lambdas * losses).sum()
