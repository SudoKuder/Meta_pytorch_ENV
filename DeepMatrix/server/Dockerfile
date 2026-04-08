# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Standalone Dockerfile for the DeepMatrix environment server.
# Build context must be the DeepMatrix/ directory:
#
#   docker build -t deepmatrix-env:latest -f DeepMatrix/server/Dockerfile DeepMatrix
#
# Or from within DeepMatrix/:
#
#   docker build -t deepmatrix-env:latest -f server/Dockerfile .

FROM python:3.11-slim AS builder

WORKDIR /app/env

# Install build tools and uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir uv

# Copy the DeepMatrix package (build context is the DeepMatrix/ directory)
COPY . /app/env

# Install dependencies into a virtual environment
RUN uv venv /app/.venv && \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable --venv /app/.venv; \
    else \
        uv sync --no-editable --venv /app/.venv; \
    fi

# Final runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy the environment code
COPY --from=builder /app/env /app/env

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Set WORKDIR to env directory so uvicorn can resolve the server module
WORKDIR /app/env

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
