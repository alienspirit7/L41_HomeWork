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

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
