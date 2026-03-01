# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install system deps needed for compiling some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch + all other deps into a venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Pre-download model weights so the container never hits HuggingFace at runtime
COPY src/ src/
COPY configs/ configs/
COPY data/nutrition_db.json data/nutrition_db.json
RUN python -c "\
import timm; timm.create_model('efficientnet_b2', pretrained=True, num_classes=0); \
print('timm efficientnet_b2 cached'); \
import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k'); \
print('CLIP ViT-B-32 cached')"


# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the pre-built venv (with cached model weights) from the builder stage
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache /root/.cache
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/          src/
COPY api/          api/
COPY scripts/      scripts/
COPY configs/      configs/
COPY ui/           ui/
COPY data/sample/  data/sample/
COPY data/nutrition_db.json  data/nutrition_db.json

# Copy model checkpoint (if it exists at build time)
COPY models/       models/

# Cloud Run injects PORT; default to 8080
ENV PORT=8080
ENV FLASK_DEBUG=0
# Suppress PyTorch NNPACK warnings (unsupported on Cloud Run CPUs)
ENV PYTORCH_DISABLE_NNPACK_LOGGING=1

EXPOSE ${PORT}

# Start the Flask API server
CMD ["sh", "-c", "python api/app.py"]
