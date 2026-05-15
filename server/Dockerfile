# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# System dep needed by docker-compose healthcheck (curl) and torch wheel runtime
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements separately for cache reuse
COPY server/requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

# Copy package preserving the `server.app` import path used in code + tests
COPY server /srv/server

# Plan 2 will copy dashboard/dist → /srv/server/app/static here.
# In Plan 1 the directory may not exist; main.py guards.

EXPOSE 8080
USER 65532:65532
CMD ["uvicorn", "server.app.main:build_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
