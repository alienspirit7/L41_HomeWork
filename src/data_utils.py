"""
Data utilities — loading, splitting, and DataLoader creation.

Also includes prepare_nutrition5k() adapter to convert raw
Nutrition5k metadata CSVs into the unified pipeline format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit

from src.dataset import FoodMacroDataset, TARGET_COLS
from src.transforms import get_train_transforms, get_eval_transforms


def compute_target_stats(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std for z-score normalisation."""
    df = pd.read_csv(csv_path)
    vals = df[TARGET_COLS].values.astype(np.float32)
    return vals.mean(axis=0), vals.std(axis=0)


def split_by_dish(csv_path: str, cfg: dict) -> tuple[pd.DataFrame, ...]:
    """Split data by dish_id to prevent leakage.

    Returns (train_df, val_df, test_df).
    """
    df = pd.read_csv(csv_path)
    groups = df["dish_id"] if "dish_id" in df.columns else df.index

    # First split: train+val vs test
    gss1 = GroupShuffleSplit(
        n_splits=1, test_size=cfg["test_split"], random_state=42,
    )
    trainval_idx, test_idx = next(gss1.split(df, groups=groups))

    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Second split: train vs val
    groups_tv = (
        df_trainval["dish_id"]
        if "dish_id" in df_trainval.columns
        else df_trainval.index
    )
    relative_val = cfg["val_split"] / (1 - cfg["test_split"])
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=relative_val, random_state=42,
    )
    train_idx, val_idx = next(gss2.split(df_trainval, groups=groups_tv))

    return (
        df_trainval.iloc[train_idx].reset_index(drop=True),
        df_trainval.iloc[val_idx].reset_index(drop=True),
        df_test,
    )


def create_dataloaders(
    cfg: dict,
    target_mean: np.ndarray | None = None,
    target_std: np.ndarray | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train / val / test DataLoaders from config."""
    csv_path = cfg["data_csv"]
    data_root = cfg.get("data_root", ".")

    train_df, val_df, test_df = split_by_dish(csv_path, cfg)

    # Save temp split CSVs
    for name, sub_df in [("train", train_df), ("val", val_df),
                         ("test", test_df)]:
        sub_df.to_csv(f"data/{name}_split.csv", index=False)

    loaders = []
    for name, sub_df, is_train in [
        ("train", train_df, True),
        ("val", val_df, False),
        ("test", test_df, False),
    ]:
        tmp_csv = f"data/{name}_split.csv"
        tfm = (get_train_transforms(cfg) if is_train
               else get_eval_transforms(cfg))
        ds = FoodMacroDataset(
            tmp_csv, data_root=data_root,
            transform=tfm,
            target_mean=target_mean, target_std=target_std,
        )
        loaders.append(DataLoader(
            ds, batch_size=cfg["batch_size"],
            shuffle=is_train, num_workers=cfg["num_workers"],
            pin_memory=True,
        ))

    return tuple(loaders)
