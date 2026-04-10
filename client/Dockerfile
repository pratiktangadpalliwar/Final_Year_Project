FROM python:3.11-slim

LABEL maintainer="fl-project"
LABEL description="FL Bank Node Client — drop CSV to trigger training"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Create data directories
RUN mkdir -p /data/input /data/processed /data/error /tmp/fl-client

# Health check — verifies watcher process is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD pgrep -f watcher.py || exit 1

# Volumes — bank operator mounts /data/input and drops CSV here
VOLUME ["/data/input", "/data/processed", "/data/error"]

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POLL_INTERVAL_SECONDS=10 \
    LOCAL_EPOCHS=10 \
    BATCH_SIZE=512 \
    LEARNING_RATE=0.001 \
    INPUT_DIM=19

# Watcher starts automatically — no manual command needed
CMD ["python", "watcher.py"]
