#!/bin/bash
# Start the original Tutel entrypoint script in the background.
# This script is responsible for starting the Tutel model serving and OpenWebUI.
# We pass the arguments that would typically be used to start the web UI and load the model.
# The `&` puts it in the background.
# Set a simple API key for OpenWebUI for internal use. This will be used by the runpod_handler.py.
# OpenWebUI documentation suggests that API key authentication might be off by default,
# but it's safer to explicitly set a key for internal communication if it's enabled.
# If OpenWebUI is configured to require an API key, this environment variable will provide it.
# The actual key can be anything, as it's for internal communication within the secure RunPod environment.
export WEBUI_SECRET_KEY="rp-tutel-internal-key"

# Check if ./openai/gpt-oss-120b exists as a symlink, if not create it
if [ ! -L "./openai/gpt-oss-120b" ]; then
    mkdir -p ./openai
    ln -s /runpod-volume/models/gpt-oss-120b ./openai/gpt-oss-120b
fi

/opt/deepseek-tutel-accel/run.sh --serve=webui --listen_port 8000 --try_path ./openai/gpt-oss-120b & 
# Wait for OpenWebUI to become available on port 8000 with a 10-minute timeout
TIMEOUT=600 # 10 minutes
INTERVAL=5  # Check every 5 seconds
ELAPSED_TIME=0
echo "Waiting for OpenWebUI to start on port 8000..."
while ! curl -s http://localhost:8000 > /dev/null; do
  if [ $ELAPSED_TIME -ge $TIMEOUT ]; then
    echo "OpenWebUI did not start within $TIMEOUT seconds. Exiting."
    exit 1
  fi
  echo "OpenWebUI not yet available. Waiting..."
  sleep $INTERVAL
  ELAPSED_TIME=$((ELAPSED_TIME + INTERVAL))
done
echo "OpenWebUI is up and running!"
# Now, execute the command passed to the entrypoint (our RunPod handler).
# This will start our RunPod handler, which will listen for incoming jobs.
exec "$@"
