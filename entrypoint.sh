#!/bin/bash
set -e

PORT="${PORT:-8090}"
HOST="${HOST:-0.0.0.0}"

echo "[entrypoint] Starting annotation server on ${HOST}:${PORT}"
exec python /app/pipeline_stable.py --host "$HOST" --port "$PORT" --skip-spotting