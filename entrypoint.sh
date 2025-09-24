#!/bin/bash

# Set the internal API key for OpenWebUI
export WEBUI_SECRET_KEY="rp-tutel-internal-key"

# Optional: Set other environment variables if needed
# export MODEL_STARTUP_TIMEOUT=600  # 10 minutes timeout
# export MODEL_CHECK_INTERVAL=5     # Check every 5 seconds
# export LOG_LEVEL=INFO

umount /app/openai 2>/dev/null || true
rm -rf /app/openai
ln -s /runpod-volume /app/openai

# Execute the command passed to the entrypoint (our RunPod handler)
# This starts our FastAPI server which will handle starting OpenWebUI
exec "$@"
