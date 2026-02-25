"""
Response/request schemas for the Food Macro API.
"""

from dataclasses import dataclass, field, asdict


@dataclass
class PredictionResponse:
    """Response schema for /predict_meal_macros."""
    weight_g: float = 0.0
    carbs_g: float = 0.0
    protein_g: float = 0.0
    fat_g: float = 0.0
    effective_carbs_g: float = 0.0
    num_images: int = 0
    confidence: str = "normal"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EffectiveCarbsRequest:
    """Request to recalculate effective carbs."""
    carbs_g: float = 0.0
    protein_g: float = 0.0
    fat_g: float = 0.0
    alpha: float = 0.5
    beta: float = 0.1
    method: str = "linear"


@dataclass
class HealthResponse:
    """Response for /health endpoint."""
    status: str = "ok"
    model_loaded: bool = False
    backbone: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
