"""
Export trained model to ONNX format.

Usage:
    python scripts/export_onnx.py \
        --checkpoint models/best.pt \
        --config configs/default.yaml \
        --output models/food_macro_model.onnx
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from src.config import load_config
from src.model import FoodMacroModel


def main():
    parser = argparse.ArgumentParser(
        description="Export model to ONNX",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="models/food_macro.onnx")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = FoodMacroModel(cfg)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu",
                    weights_only=True)
    )
    model.eval()

    size = cfg["image_size"]
    dummy = torch.randn(1, 3, size, size)

    torch.onnx.export(
        model, dummy, args.output,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["macros"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "macros": {0: "batch_size"},
        },
    )
    print(f"ONNX model exported to {args.output}")
    print(f"Input shape:  (batch, 3, {size}, {size})")
    print("Output shape: (batch, 4)  "
          "[weight_norm, carbs_norm, protein_norm, fat_norm]")


if __name__ == "__main__":
    main()
