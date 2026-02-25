"""
Evaluation metrics for food macro estimation.

All metrics operate on de-normalised values (grams).
"""

import numpy as np

TARGET_NAMES = ["weight_g", "carbs_g", "protein_g", "fat_g"]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mean Absolute Error per target. Shape: (4,)."""
    return np.abs(y_true - y_pred).mean(axis=0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Root Mean Squared Error per target. Shape: (4,)."""
    return np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))


def mape(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0,
) -> np.ndarray:
    """Mean Absolute Percentage Error per target.

    eps avoids division by zero for targets near 0.
    """
    return (
        np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps)
    ).mean(axis=0) * 100.0


def mean_signed_error(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> np.ndarray:
    """Mean signed error (bias) per target.

    Positive = systematic over-estimation.
    Negative = systematic under-estimation.
    """
    return (y_pred - y_true).mean(axis=0)


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute all metrics and return as nested dict.

    Returns:
        {metric_name: {target_name: value, ...}, ...}
    """
    results = {}
    for metric_fn, label in [
        (mae, "mae"), (rmse, "rmse"),
        (mape, "mape"), (mean_signed_error, "bias"),
    ]:
        vals = metric_fn(y_true, y_pred)
        results[label] = {
            name: round(float(v), 4)
            for name, v in zip(TARGET_NAMES, vals)
        }
    return results


def format_metrics(metrics: dict) -> str:
    """Format metrics dict into a readable table string."""
    lines = [f"{'Metric':<8} {'Weight':>10} {'Carbs':>10} "
             f"{'Protein':>10} {'Fat':>10}"]
    lines.append("-" * 52)
    for metric_name, vals in metrics.items():
        row = f"{metric_name:<8}"
        for target in TARGET_NAMES:
            row += f" {vals[target]:>10.2f}"
        lines.append(row)
    return "\n".join(lines)
