import runpod
from runpod.serverless.utils.rp_validator import validate
import requests
import os
import json
import time

# The internal OpenWebUI is expected to run on port 8000 within the container
# and exposes an OpenAI-compatible API at /api/chat/completions
OPENWEBUI_INTERNAL_API_URL = os.getenv("OPENWEBUI_INTERNAL_API_URL", "http://localhost:8000/api/chat/completions")

# Retrieve the internal API key set in the entrypoint.sh
OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY")

# Schema for validating the input payload
INPUT_SCHEMA = {
    'model': {
        'type': str,
        'required': True
    },
    'messages': {
        'type': list,
        'required': True,
        'min_length': 1
    },
    'temperature': {
        'type': float,
        'required': False,
        'default': 1.0,
        'constraints': [lambda temp: 0.0 <= temp <= 2.0, 'Temperature must be between 0.0 and 2.0']
    },
    'max_tokens': {
        'type': int,
        'required': False,
        'default': 32768,
        'constraints': [lambda tokens: tokens > 0, 'Max tokens must be greater than 0']
    },
    'stream': {
        'type': bool,
        'required': False,
        'default': False
    }
}

def handler(job):
    job_input = job['input']

    # Validate the input against the schema
    validated_input = validate(job_input, INPUT_SCHEMA)
    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    # Prepare headers, including the API key if available
    headers = {"Content-Type": "application/json"}
    if OPENWEBUI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENWEBUI_API_KEY}"

    # Forward the request directly to the OpenWebUI API
    try:
        if job_input.get('stream', False):
            # For streaming, we need to return the raw response content as RunPod will handle the SSE parsing.
            # The `stream=True` in requests.post is crucial for this.
            response = requests.post(
                OPENWEBUI_INTERNAL_API_URL,
                json=job_input,
                headers=headers,
                stream=True,
                timeout=600
            )
            response.raise_for_status()
            # RunPod expects the raw text content for streaming.
            return {"output": response.text, "is_stream": True}
        else:
            # Handle non-streaming response
            response = requests.post(
                OPENWEBUI_INTERNAL_API_URL,
                json=job_input,
                headers=headers,
                timeout=600
            )
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to communicate with internal OpenWebUI service: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred in handler: {e}"}

runpod.serverless.start({"handler": handler})
