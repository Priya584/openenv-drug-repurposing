# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# git is required for VCS-based dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy environment code
COPY . /app/env
WORKDIR /app/env

# Ensure uv is available
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies
# Explicitly delete any stale uv.lock that may be cached on the Hugging Face space
RUN rm -f uv.lock

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-editable

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable

# ── Runtime stage ─────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

# curl is needed by the HEALTHCHECK in this stage
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy virtualenv and source from builder
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env       /app/env

# Ensure inference.py is at the project root (spec requirement)
COPY inference.py /app/inference.py

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Default env vars — override at runtime via docker run -e or HF Space secrets
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV ENV_BASE_URL="http://localhost:8000"
ENV DRUG_TASK="repurpose"
# HF_TOKEN must be supplied at runtime — no default for security reasons

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]