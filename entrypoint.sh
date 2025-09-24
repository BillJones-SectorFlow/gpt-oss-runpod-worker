#!/bin/bash
echo "=== Starting entrypoint.sh ==="
echo "Setting WEBUI_SECRET_KEY..."
export WEBUI_SECRET_KEY="rp-tutel-internal-key"

echo "Current working directory: $(pwd)"
echo "=== Entrypoint.sh completed, starting main command ==="
echo "Executing: $@"

exec "$@"
