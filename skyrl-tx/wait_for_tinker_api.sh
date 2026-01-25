#!/bin/bash
# Usage: wait_for_tinker_api.sh <url> <timeout_seconds>
# Waits for Tinker API to be ready by polling the healthz endpoint

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <url> <timeout_seconds>"
    echo "  url: Base URL of Tinker API (e.g., http://localhost:8080)"
    echo "  timeout_seconds: Maximum time to wait"
    exit 1
fi

URL=$1
TIMEOUT=$2

if ! [[ "$TIMEOUT" =~ ^[0-9]+$ ]]; then
    echo "ERROR: timeout_seconds must be a positive integer, got: $TIMEOUT"
    exit 1
fi

HEALTH_ENDPOINT="$URL/api/v1/healthz"
echo "Waiting for Tinker API at $URL..."

API_READY=0
for COUNTER in $(seq 1 $TIMEOUT); do
    if curl -s "$HEALTH_ENDPOINT" 2>/dev/null | grep -q '"status":"ok"'; then
        echo "Tinker API ready after $COUNTER seconds"
        API_READY=1
        break
    fi
    if [ $((COUNTER % 60)) -eq 0 ]; then
        echo "Still waiting... ($COUNTER seconds)"
    fi
    sleep 1
done

if [ "$API_READY" -ne 1 ]; then
    echo "ERROR: Tinker API timed out after $TIMEOUT seconds"
    exit 1
fi
