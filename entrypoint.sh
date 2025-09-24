#!/bin/bash
echo "=== Starting entrypoint.sh ==="
echo "Setting WEBUI_SECRET_KEY..."
export WEBUI_SECRET_KEY="rp-tutel-internal-key"

echo "Checking if /app/openai is a mount point..."
if mountpoint -q /app/openai 2>/dev/null; then
    echo "/app/openai is mounted, unmounting..."
    umount /app/openai
    echo "Unmount completed"
else
    echo "/app/openai is not a mount point"
fi

echo "Removing /app/openai directory if it exists..."
rm -rf /app/openai
echo "Directory removal completed"

echo "Creating symbolic link from /app/openai to /runpod-volume..."
ln -s /runpod-volume /app/openai
echo "Symbolic link created"

echo "Verifying symbolic link..."
if [ -L /app/openai ]; then
    echo "SUCCESS: /app/openai is a symbolic link pointing to: $(readlink /app/openai)"
else
    echo "ERROR: /app/openai is not a symbolic link"
fi

echo "Checking if /runpod-volume exists and is accessible..."
if [ -d /runpod-volume ]; then
    echo "SUCCESS: /runpod-volume exists"
    echo "Contents of /runpod-volume: $(ls -la /runpod-volume 2>/dev/null || echo 'Cannot list contents')"
else
    echo "ERROR: /runpod-volume does not exist"
fi

echo "Current working directory: $(pwd)"
echo "=== Entrypoint.sh completed, starting main command ==="
echo "Executing: $@"

exec "$@"
