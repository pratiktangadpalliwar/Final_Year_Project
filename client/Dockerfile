# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /work

# System: no curl needed in client (no health endpoint), but compose's healthcheck
# may need it later.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY client/requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

# Client imports `from server.app.dp_engine import gaussian_sigma`.
# Plan 3 will move this to a shared package; for now copy the server tree too.
COPY server /work/server
COPY client /work/client

# /work/data/bank.csv is supplied by the init container at runtime.
USER 65532:65532
CMD ["python", "-m", "client.app.main"]
