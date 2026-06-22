#!/bin/bash
set -e

IMAGE_NAME="predictive-maintenance"

echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:latest .
echo "Build completed."