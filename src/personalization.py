"""
Personalization scaffold — user meal store and calibration.

Provides:
  - UserMealStore: persist user-logged meals as JSON
  - CalibrationLayer: per-user scale+bias on model outputs
  - Fine-tuning utility for adapter layers
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime


class UserMealStore:
    """Append-only JSON store for a user's meal logs."""

    def __init__(self, user_id: str, store_dir: str = "data/users"):
        self.user_id = user_id
        self.path = Path(store_dir) / f"{user_id}_meals.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.meals = self._load()

    def _load(self) -> list[dict]:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return []

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self.meals, f, indent=2)

    def add_meal(
        self,
        image_paths: list[str],
        predicted: dict,
        corrected: dict | None = None,
    ) -> None:
        """Log a meal with predictions and optional corrections."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "image_paths": image_paths,
            "predicted": predicted,
            "corrected": corrected,
        }
        self.meals.append(entry)
        self._save()

    def get_corrections(self) -> list[dict]:
        """Return only meals where user provided corrections."""
        return [m for m in self.meals if m.get("corrected")]


class CalibrationLayer(nn.Module):
    """Per-user scale + bias on model outputs.

    output_calibrated = output * scale + bias

    Learned from user-corrected meals via simple regression.
    """

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(4))
        self.bias = nn.Parameter(torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.bias

    def fit_from_corrections(
        self, predicted: np.ndarray, actual: np.ndarray,
        lr: float = 0.01, steps: int = 200,
    ) -> None:
        """Fit scale+bias from (predicted, actual) pairs."""
        pred_t = torch.tensor(predicted, dtype=torch.float32)
        actual_t = torch.tensor(actual, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.train()
        for _ in range(steps):
            optimizer.zero_grad()
            out = self(pred_t)
            loss = loss_fn(out, actual_t)
            loss.backward()
            optimizer.step()
        self.eval()
