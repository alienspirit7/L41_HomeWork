#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Download Nutrition5k dataset (metadata + overhead RGB images)
# ──────────────────────────────────────────────────────────────
# This script downloads ONLY the metadata CSVs and overhead RGB
# images from the Nutrition5k Google Cloud bucket.
# Full dataset is ~181 GB; we download ~5-10 GB (images only).
#
# Prerequisites: gsutil (installed via pip if missing)
# Usage: bash scripts/download_data.sh
# ──────────────────────────────────────────────────────────────

set -e

BUCKET="gs://nutrition5k_dataset/nutrition5k_dataset"
DATA_DIR="data/nutrition5k"

mkdir -p "$DATA_DIR/metadata"
mkdir -p "$DATA_DIR/imagery"

# ── Install gsutil if not available ───────────────────────────
if ! command -v gsutil &> /dev/null; then
    echo "gsutil not found. Installing via pip..."
    pip install gsutil
fi

# ── Download metadata CSVs ────────────────────────────────────
echo "=== Downloading metadata CSVs ==="
gsutil -m cp -r "${BUCKET}/metadata/*" "$DATA_DIR/metadata/"
echo "Metadata downloaded."

# ── Download dish ID splits ───────────────────────────────────
echo "=== Downloading train/test splits ==="
gsutil -m cp -r "${BUCKET}/dish_ids/" "$DATA_DIR/dish_ids/"
echo "Splits downloaded."

# ── Download overhead RGB images ──────────────────────────────
echo "=== Downloading overhead RGB images ==="
echo "This may take a while (~5-10 GB)..."
gsutil -m cp -r "${BUCKET}/imagery/realsense_overhead/" \
    "$DATA_DIR/imagery/realsense_overhead/"
echo "Images downloaded."

# ── Build unified processed CSV ──────────────────────────────
echo "=== Building processed.csv ==="
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python "$SCRIPT_DIR/prepare_nutrition5k.py"

echo ""
echo "=== Download complete ==="
echo "Data directory: $DATA_DIR"
echo "Processed CSV:  $DATA_DIR/processed.csv"
