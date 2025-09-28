# Use the official Tutel Docker image as the base
FROM tutelgroup/deepseek-671b:a100x8-chat-20250827

# Set up the working directory
WORKDIR /app

# Install Python dependencies for the handler and curl for the entrypoint script
RUN pip install requests runpod && apt-get update && apt-get install -y curl

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the RunPod handler script into the container
COPY handler.py /app/handler.py

# Copy our custom entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

RUN rm -rf /opt/deepseek-tutel-accel/llm_moe_tutel.py
COPY llm_moe_tutel.py /opt/deepseek-tutel-accel/llm_moe_tutel.py

# Expose the port for our handler OpenAI-API-Compatible endpoint
EXPOSE 80

# Set our custom script as the new entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# The default command will be to start the handler
CMD ["python", "-u", "/app/handler.py"]
