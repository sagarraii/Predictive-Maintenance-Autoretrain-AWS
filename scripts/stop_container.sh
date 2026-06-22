#!/bin/bash
set -e

CONTAINER_NAME="predictive-maintenance-app"

docker stop ${CONTAINER_NAME} 2>/dev/null || true
docker rm ${CONTAINER_NAME} 2>/dev/null || true

echo "Old container removed (if it existed)."