#!/bin/bash

# Set the internal API key for OpenWebUI
export WEBUI_SECRET_KEY="rp-tutel-internal-key"

# Optional: Set other environment variables if needed
# export MODEL_STARTUP_TIMEOUT=600  # 10 minutes timeout
# export MODEL_CHECK_INTERVAL=5     # Check every 5 seconds
# export LOG_LEVEL=INFO

mkdir -p /runpod-volume
sudo mount -t nfs -o nconnect=16 nfs.fin-03.datacrunch.io:/gpt-oss-b141abc0 /runpod-volume || echo "Could not mount NFS share"

# Execute the command passed to the entrypoint (our RunPod handler)
# This starts our FastAPI server which will handle starting OpenWebUI
exec "$@"
