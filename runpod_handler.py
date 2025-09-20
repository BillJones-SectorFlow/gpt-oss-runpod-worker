
import os
import logging
import httpx
from fastapi import FastAPI, Request, Response, status, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# The internal OpenWebUI is expected to run on port 8000 within the container
# and exposes an OpenAI-compatible API at /v1/chat/completions
OPENWEBUI_INTERNAL_API_URL = os.getenv("OPENWEBUI_INTERNAL_API_URL", "http://localhost:8000/v1/chat/completions")

# Retrieve the internal API key set in the entrypoint.sh
WEBUI_SECRET_KEY = os.getenv("WEBUI_SECRET_KEY")

# --- FastAPI Application Setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI application startup.")
    # Any startup tasks can go here. For now, just logging.
    yield
    logger.info("FastAPI application shutdown.")
    # Any shutdown tasks can go here.

app = FastAPI(title="OpenWebUI Proxy Load Balancer", version="1.0.0", lifespan=lifespan)

# --- Pydantic Models for Request Validation ---

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0)
    top_k: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=32768, gt=0)
    stream: bool = False
    reasoning_effort: str = "high"

# --- Endpoints ---

@app.get("/ping")
async def ping():
    """Health check endpoint."""
    logger.info("Received /ping request.")
    return Response(status_code=status.HTTP_200_OK)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, chat_request: ChatCompletionRequest):
    """Proxies chat completion requests to the OpenWebUI instance."""
    request_id = request.headers.get("X-Request-ID", "unknown")
    logger.info(f"[{request_id}] Received chat completion request for model: {chat_request.model}")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {WEBUI_SECRET_KEY}"
    }

    try:
        async with httpx.AsyncClient() as client:
            # Use the Pydantic model's dict() method to get the payload for the proxy request
            # This ensures default values are included if not provided in the original request
            openai_payload = chat_request.dict(exclude_unset=True)

            if chat_request.stream:
                # For streaming requests, forward the stream directly
                req = client.build_request("POST", OPENWEBUI_INTERNAL_API_URL, json=openai_payload, headers=headers, timeout=None)
                r = await client.send(req, stream=True)
                r.raise_for_status()
                logger.info(f"[{request_id}] Streaming response from OpenWebUI.")
                return StreamingResponse(r.aiter_bytes(), media_type=r.headers.get("Content-Type", "text/event-stream"))
            else:
                # For non-streaming requests, get the JSON response
                response = await client.post(OPENWEBUI_INTERNAL_API_URL, json=openai_payload, headers=headers, timeout=None)
                response.raise_for_status()
                logger.info(f"[{request_id}] Non-streaming response from OpenWebUI.")
                return JSONResponse(content=response.json())

    except ValidationError as e:
        logger.error(f"[{request_id}] Request validation error: {e.errors()}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "Invalid request payload",
                    "type": "validation_error",
                    "code": "invalid_payload",
                    "errors": e.errors()
                }
            }
        )
    except httpx.RequestError as e:
        logger.error(f"[{request_id}] Error proxying request to OpenWebUI: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Error communicating with OpenWebUI: {e}",
                    "type": "proxy_error",
                    "code": "openwebui_connection_error"
                }
            }
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"[{request_id}] OpenWebUI returned an error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail={
                "error": {
                    "message": f"OpenWebUI returned an error: {e.response.text}",
                    "type": "openwebui_error",
                    "code": f"http_status_{e.response.status_code}"
                }
            }
        )
    except Exception as e:
        logger.error(f"[{request_id}] An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"An unexpected error occurred: {e}",
                    "type": "unexpected_error",
                    "code": "internal_server_error"
                }
            }
        )

if __name__ == "__main__":
    # Get ports from environment variables
    port = int(os.getenv("PORT", 80))
    logger.info(f"Starting server on port {port}")

    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )

