"""
Warsaw Method (FPU) Bolus Recommendation Engine.

Calculates Fat-Protein Units and generates dual-wave bolus
recommendations for insulin pump users.

Methods:
  - linear:  effective_carbs = carbs + α·protein + β·fat
  - warsaw:  Full FPU engine with extension duration + dual wave
"""


def compute_fpu(protein: float, fat: float) -> float:
    """Calculate Fat-Protein Units.

    1 FPU = 100 kcal from protein + fat.
    Protein: 4 kcal/g, Fat: 9 kcal/g.
    """
    total_cal = protein * 4.0 + fat * 9.0
    return round(total_cal / 100.0, 2)


def compute_equivalent_carbs(
    fpu: float, fpu_to_carb: float = 10.0,
) -> float:
    """Convert FPU to equivalent carbs.

    Default: 1 FPU = 10g carbs.
    Adjustable: 7g or 5g based on personal CGM trends.
    """
    return round(fpu * fpu_to_carb, 1)


def get_extension_duration(fpu: float) -> int:
    """Extension duration (hours) based on FPU count.

    | FPU   | Duration |
    |-------|----------|
    | ≤1    | 3 hours  |
    | ≤2    | 4 hours  |
    | ≤3    | 5 hours  |
    | >3    | 8 hours  |
    """
    if fpu <= 1:
        return 3
    elif fpu <= 2:
        return 4
    elif fpu <= 3:
        return 5
    return 8


def compute_bolus_recommendation(
    carbs: float,
    protein: float,
    fat: float,
    icr: float = 10.0,
    fpu_to_carb: float = 10.0,
    activity_reduction: bool = False,
) -> dict:
    """Full Warsaw Method bolus recommendation.

    Args:
        carbs:    grams of carbohydrates
        protein:  grams of protein
        fat:      grams of fat
        icr:      Insulin-to-Carb Ratio (1 unit per N grams)
        fpu_to_carb: grams of carb equivalent per FPU
        activity_reduction: if True, halve EC (post-meal activity)

    Returns:
        Dict with full bolus recommendation.
    """
    fpu = compute_fpu(protein, fat)
    ec = compute_equivalent_carbs(fpu, fpu_to_carb)

    if activity_reduction:
        ec = round(ec * 0.5, 1)

    total_active = round(carbs + ec, 1)
    duration = get_extension_duration(fpu)

    # Insulin units
    immediate_units = round(carbs / icr, 1) if icr > 0 else 0
    extended_units = round(ec / icr, 1) if icr > 0 else 0
    total_units = round(immediate_units + extended_units, 1)

    # Percentages for dual-wave
    if total_active > 0:
        immediate_pct = round(carbs / total_active * 100)
        extended_pct = 100 - immediate_pct
    else:
        immediate_pct, extended_pct = 100, 0

    return {
        "fpu": fpu,
        "equivalent_carbs_g": ec,
        "total_active_carbs_g": total_active,
        "immediate_carbs_g": round(carbs, 1),
        "extended_carbs_g": ec,
        "extension_duration_hours": duration,
        "total_insulin_units": total_units,
        "immediate_units": immediate_units,
        "extended_units": extended_units,
        "immediate_pct": immediate_pct,
        "extended_pct": extended_pct,
        "strategy": "dual_wave" if ec > 0 else "normal",
        "activity_reduction_applied": activity_reduction,
    }


# ── Backward-compatible wrappers ──────────────────────────────


def compute_effective_carbs(
    carbs: float,
    protein: float,
    fat: float,
    alpha: float = 0.5,
    beta: float = 0.1,
    method: str = "linear",
) -> float:
    """Compute effective carbs (backward-compatible).

    Args:
        method: 'linear' or 'warsaw'
    """
    if method == "linear":
        return carbs + alpha * protein + beta * fat
    elif method == "warsaw":
        fpu = compute_fpu(protein, fat)
        ec = compute_equivalent_carbs(fpu)
        return carbs + ec
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


def bolus_recommendation_from_config(
    carbs: float, protein: float, fat: float, cfg: dict,
) -> dict:
    """Full bolus recommendation using config parameters."""
    return compute_bolus_recommendation(
        carbs, protein, fat,
        icr=cfg.get("icr", 10.0),
        fpu_to_carb=cfg.get("fpu_to_carb_ratio", 10.0),
        activity_reduction=cfg.get("activity_reduction", False),
    )
