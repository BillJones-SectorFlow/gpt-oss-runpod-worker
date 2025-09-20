import os
import logging
import requests
import runpod
from runpod.serverless.utils.rp_validator import validate

# Configure logging
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger(__name__)

# The internal OpenWebUI is expected to run on port 8000 within the container
# and exposes an OpenAI-compatible API at /v1/chat/completions
OPENWEBUI_INTERNAL_API_URL = os.getenv("OPENWEBUI_INTERNAL_API_URL", "http://localhost:8000/v1/chat/completions")

# Retrieve the internal API key set in the entrypoint.sh
WEBUI_SECRET_KEY = os.getenv("WEBUI_SECRET_KEY")

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
    log.info(f"Received job: {job['id']}")
    log.debug(f"Raw job input: {job_input}")

    # Validate the input against the schema
    validated_input = validate(job_input, INPUT_SCHEMA)
    if 'errors' in validated_input:
        log.error(f"Validation errors: {validated_input['errors']}")
        return {"error": validated_input['errors']}
    
    # The validated input is now directly the OpenAI API payload
    openai_payload = validated_input['validated_input']
    log.debug(f"OpenAI payload extracted: {openai_payload}")

    # Prepare headers, including the API key if available
    headers = {"Content-Type": "application/json"}
    if OPENWEBUI_API_KEY:
        headers["Authorization"] = f"Bearer {WEBUI_SECRET_KEY}"

    # Forward the request directly to the OpenWebUI API
    try:
        log.info("Forwarding request to OpenWebUI...")
        if openai_payload.get('stream', False):
            # For streaming, we need to return a generator that yields the chunks.
            # RunPod's serverless worker will handle the streaming back to the client.
            def stream_generator():
                with requests.post(
                    OPENWEBUI_INTERNAL_API_URL,
                    json=openai_payload,
                    headers=headers,
                    stream=True,
                    timeout=600
                ) as response:
                    response.raise_for_status()
                    log.info("Streaming response from OpenWebUI...")
                    for chunk in response.iter_content(chunk_size=8192):
                        log.debug(f"Yielding chunk: {chunk}")
                        yield chunk
            return stream_generator()
        else:
            # Handle non-streaming response
            response = requests.post(
                OPENWEBUI_INTERNAL_API_URL,
                json=openai_payload,
                headers=headers,
                timeout=600
            )
            response.raise_for_status()  # Raise an exception for HTTP errors
            result = response.json()
            log.info("Received non-streaming response from OpenWebUI.")
            log.debug(f"OpenWebUI response: {result}")
            # The key insight from the vLLM worker is that the final output needs to be a dictionary
            # with a specific structure. For non-streaming, it should be `{"output": result}`.
            return {"output": result}

    except requests.exceptions.RequestException as e:
        log.error(f"Failed to communicate with internal OpenWebUI service: {e}")
        return {"error": f"Failed to communicate with internal OpenWebUI service: {e}"}
    except Exception as e:
        log.error(f"An unexpected error occurred in handler: {e}")
        return {"error": f"An unexpected error occurred in handler: {e}"}

runpod.serverless.start({"handler": handler})
