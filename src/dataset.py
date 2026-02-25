"""
FoodMacroDataset — PyTorch Dataset for food macro regression.

Reads a CSV with columns:
  dish_id, image_path, weight_g, carbs_g, protein_g, fat_g

Returns (image_tensor, target_tensor) where target_tensor is
[weight, carbs, protein, fat] optionally z-score normalised.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np


TARGET_COLS = ["weight_g", "carbs_g", "protein_g", "fat_g"]


class FoodMacroDataset(Dataset):
    """Dataset for food images with macro targets."""

    def __init__(
        self,
        csv_path: str,
        data_root: str = ".",
        transform=None,
        target_mean: np.ndarray | None = None,
        target_std: np.ndarray | None = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.transform = transform
        self.target_mean = target_mean
        self.target_std = target_std

        # Validate required columns
        missing = [c for c in TARGET_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        self.targets = self.df[TARGET_COLS].values.astype(np.float32)

        # Apply z-score normalisation if stats provided
        if self.target_mean is not None and self.target_std is not None:
            self.targets = (
                (self.targets - self.target_mean) / (self.target_std + 1e-8)
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.data_root / row["image_path"]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        targets = torch.tensor(self.targets[idx], dtype=torch.float32)
        return img, targets

    @property
    def dish_ids(self) -> np.ndarray:
        """Return dish IDs for splitting by dish."""
        if "dish_id" in self.df.columns:
            return self.df["dish_id"].values
        return np.arange(len(self.df))
