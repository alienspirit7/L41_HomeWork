"""
Flask API for Food Macro Estimation.

Endpoints:
  GET  /health                → service health check
  POST /predict_meal_macros   → predict macros from images
  POST /effective_carbs       → recalculate effective carbs
"""

import io
import sys
import os
import tempfile
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config
from src.inference import predict_meal
from src.effective_carbs import compute_effective_carbs
from api.schemas import PredictionResponse, HealthResponse


def create_app(config_path: str = "configs/default.yaml"):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    cfg = load_config(config_path)
    checkpoint = cfg.get("checkpoint_dir", "models") + "/best.pt"

    @app.route("/health", methods=["GET"])
    def health():
        resp = HealthResponse(
            status="ok",
            model_loaded=os.path.exists(checkpoint),
            backbone=cfg["backbone"],
        )
        return jsonify(resp.to_dict())

    @app.route("/predict_meal_macros", methods=["POST"])
    def predict():
        files = request.files.getlist("images")
        if not files:
            return jsonify({"error": "No images provided"}), 400
        if len(files) > 3:
            return jsonify({"error": "Max 3 images allowed"}), 400

        # Save uploaded images to temp files
        temp_paths = []
        try:
            for f in files:
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False,
                )
                f.save(tmp.name)
                temp_paths.append(tmp.name)

            ckpt = checkpoint if os.path.exists(checkpoint) else None
            result = predict_meal(temp_paths, cfg, checkpoint=ckpt)

            resp = PredictionResponse(**result)
            return jsonify(resp.to_dict())
        finally:
            for p in temp_paths:
                os.unlink(p)

    @app.route("/effective_carbs", methods=["POST"])
    def recalc_effective_carbs():
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        ec = compute_effective_carbs(
            carbs=data.get("carbs_g", 0),
            protein=data.get("protein_g", 0),
            fat=data.get("fat_g", 0),
            alpha=data.get("alpha", 0.5),
            beta=data.get("beta", 0.1),
            method=data.get("method", "linear"),
        )
        return jsonify({"effective_carbs_g": round(ec, 1)})

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
