# Use the official Tutel Docker image as the base
FROM tutelgroup/deepseek-671b:a100x8-chat-20250827

# Set up the working directory
WORKDIR /app

# Install Python dependencies for the RunPod handler and curl for the entrypoint script
RUN pip install requests runpod && apt-get update && apt-get install -y curl

# Copy the RunPod handler script into the container
COPY runpod_handler.py /app/runpod_handler.py

# Copy our custom entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Expose the port for OpenWebUI (which our handler will proxy to)
EXPOSE 8000 80

# Set our custom script as the new entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# The default command will be to start the RunPod handler
CMD ["python", "-u", "/app/runpod_handler.py"]
