"""
Effective carbs calculation.

Provides configurable formulas to convert raw macros into
"effective carbs" for insulin bolus calculation.

Default (linear):
  effective_carbs = carbs + alpha * protein + beta * fat

Warsaw method:
  effective_carbs = carbs + (protein * 4 + fat * 9) / 10
"""


def compute_effective_carbs(
    carbs: float,
    protein: float,
    fat: float,
    alpha: float = 0.5,
    beta: float = 0.1,
    method: str = "linear",
) -> float:
    """Compute effective carbs from macronutrients.

    Args:
        carbs:   grams of carbohydrates
        protein: grams of protein
        fat:     grams of fat
        alpha:   protein coefficient (linear method)
        beta:    fat coefficient (linear method)
        method:  'linear' or 'warsaw'

    Returns:
        Effective carbs value (float).
    """
    if method == "linear":
        return carbs + alpha * protein + beta * fat
    elif method == "warsaw":
        # Warsaw method: protein & fat converted to energy,
        # then divided by carb energy density (~10 kcal/g)
        return carbs + (protein * 4.0 + fat * 9.0) / 10.0
    else:
        raise ValueError(f"Unknown method: {method}")


def effective_carbs_from_config(
    carbs: float, protein: float, fat: float, cfg: dict,
) -> float:
    """Compute effective carbs using config parameters."""
    return compute_effective_carbs(
        carbs, protein, fat,
        alpha=cfg.get("effective_carbs_alpha", 0.5),
        beta=cfg.get("effective_carbs_beta", 0.1),
        method=cfg.get("effective_carbs_method", "linear"),
    )
