#!/bin/bash
set -e

IMAGE_NAME="predictive-maintenance"
CONTAINER_NAME="predictive-maintenance-app"

docker run -d \
    --name ${CONTAINER_NAME} \
    --restart unless-stopped \
    -p 5000:5000 \
    ${IMAGE_NAME}:latest

echo "Container started."