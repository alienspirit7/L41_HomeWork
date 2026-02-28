"""
Prediction script — run inference from command line.

Usage:
    python scripts/predict.py \
        --images meal1.jpg meal2.jpg \
        --config configs/default.yaml \
        --checkpoint models/best.pt
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config
from src.inference import predict_meal


def print_bolus_recommendation(bolus: dict) -> None:
    """Pretty-print the bolus recommendation."""
    print("\n" + "=" * 50)
    print("  💉 BOLUS RECOMMENDATION (Warsaw Method)")
    print("=" * 50)
    print(f"  Fat-Protein Units (FPU):    {bolus['fpu']}")
    print(f"  Equivalent Carbs (EC):      {bolus['equivalent_carbs_g']}g")
    print(f"  Total Active Carbs:         "
          f"{bolus['total_active_carbs_g']}g")
    print("-" * 50)
    print(f"  Strategy:                   {bolus['strategy']}")
    print(f"  Immediate (carbs):          "
          f"{bolus['immediate_carbs_g']}g  "
          f"({bolus['immediate_pct']}%)")
    print(f"  Extended  (fat+protein):    "
          f"{bolus['extended_carbs_g']}g  "
          f"({bolus['extended_pct']}%)")
    print(f"  Extension duration:         "
          f"{bolus['extension_duration_hours']} hours")
    print("-" * 50)
    print(f"  Immediate insulin:          "
          f"{bolus['immediate_units']}u")
    print(f"  Extended insulin:           "
          f"{bolus['extended_units']}u")
    print(f"  Total insulin:              "
          f"{bolus['total_insulin_units']}u")
    if bolus.get("activity_reduction_applied"):
        print("  ⚠️  Activity reduction applied (EC halved)")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Predict meal macros from images",
    )
    parser.add_argument(
        "--images", nargs="+", required=True,
        help="1-3 image paths",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--device", default="cpu",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = predict_meal(
        args.images, cfg,
        checkpoint=args.checkpoint,
        device=args.device,
    )

    # Print macro predictions
    bolus = result.pop("bolus_recommendation", None)
    print(json.dumps(result, indent=2))

    # Print bolus recommendation
    if bolus:
        print_bolus_recommendation(bolus)


if __name__ == "__main__":
    main()
