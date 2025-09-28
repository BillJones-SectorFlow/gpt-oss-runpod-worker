#!/bin/bash
echo "=== Starting entrypoint.sh ==="
echo "Setting TUTEL_INTERNAL_ONLY_KEY..."
export TUTEL_INTERNAL_ONLY_KEY="tutel-internal-key"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/x86_64-linux-gnu
export TORCHDYNAMO_VERBOSE=1
export NCCL_DEBUG=INFO
export PYTHONDONTWRITEBYTECODE=1

echo "Current working directory: $(pwd)"
echo "=== Entrypoint.sh completed, starting main command ==="
echo "Executing: $@"

exec "$@"
