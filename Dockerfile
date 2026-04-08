# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Dockerfile for Hugging Face Spaces deployment.
# Builds a self-contained image for the DeepMatrix FastAPI environment server.

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the DeepMatrix package source
COPY DeepMatrix/ ./DeepMatrix/

# Make the DeepMatrix package importable from /app
ENV PYTHONPATH=/app

# Expose the application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "DeepMatrix.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
