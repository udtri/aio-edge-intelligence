# =============================================================================
# AIO Sensor Intelligence — Inference Server
# Multi-stage Dockerfile supporting both CPU and CUDA deployments.
#
# Build examples:
#   CPU (default):  docker build -t inference-server .
#   CUDA:           docker build --build-arg RUNTIME=cuda -t inference-server:cuda .
# =============================================================================

ARG RUNTIME=cpu

# ---------- Stage: base (CPU) ------------------------------------------------
FROM python:3.11-slim AS base-cpu

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl && \
    rm -rf /var/lib/apt/lists/*

# ---------- Stage: base (CUDA) -----------------------------------------------
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS base-cuda

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        build-essential \
        curl && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

# ---------- Stage: final (selected by ARG) ------------------------------------
FROM base-${RUNTIME} AS final

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
