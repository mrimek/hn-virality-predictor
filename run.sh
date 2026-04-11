#!/bin/bash
# Start the HN Virality Predictor web server
# Usage: ./run.sh [port]   (default: 8000)

PORT=${1:-8000}
cd "$(dirname "$0")"
echo "Starting HN Virality Predictor on http://localhost:$PORT"
uvicorn api:app --host 0.0.0.0 --port "$PORT" --reload
