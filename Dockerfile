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


# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the pre-built venv from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/          src/
COPY api/          api/
COPY scripts/      scripts/
COPY configs/      configs/
COPY ui/           ui/
COPY data/sample/  data/sample/

# Copy model checkpoint (if it exists at build time)
COPY models/       models/

# Cloud Run injects PORT; default to 8080
ENV PORT=8080

EXPOSE ${PORT}

# Start the Flask API server
CMD ["sh", "-c", "python api/app.py"]
