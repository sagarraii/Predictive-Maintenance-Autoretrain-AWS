#!/bin/bash

echo "Running containers:"
docker ps

echo ""
echo "Application logs:"
docker logs predictive-maintenance-app --tail 20