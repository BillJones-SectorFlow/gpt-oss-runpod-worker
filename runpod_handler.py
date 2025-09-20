import os
import logging
import httpx
import uvicorn
from fastapi import FastAPI, Request, Response, status, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from typing import List, Dict, Any, Optional, Union, Literal
from contextlib import asynccontextmanager

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
# The internal OpenWebUI is expected to run on port 8000 within the container
# and exposes an OpenAI-compatible API at /v1/chat/completions
OPENWEBUI_INTERNAL_API_URL = os.getenv(
    "OPENWEBUI_INTERNAL_API_URL", "http://localhost:8000/v1/chat/completions"
)

# Retrieve the internal API key set in the entrypoint.sh
WEBUI_SECRET_KEY = os.getenv("WEBUI_SECRET_KEY")

# ------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI application startup.")
    yield
    logger.info("FastAPI application shutdown.")

app = FastAPI(title="OpenWebUI Proxy Load Balancer", version="1.0.0", lifespan=lifespan)

# ------------------------------------------------------------
# Pydantic Models (OpenAI-compatible and permissive)
# ------------------------------------------------------------

class ImageURL(BaseModel):
    url: str
    # Keep it permissive: some providers accept detail={"low"|"high"|"auto"}
    detail: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class InputAudio(BaseModel):
    data: str            # base64
    format: Optional[str] = None  # e.g., "wav", "mp3"
    model_config = ConfigDict(extra="allow")


class ContentPart(BaseModel):
    # Common OpenAI-style parts; keep permissive to avoid schema churn
    type: Literal["text", "image_url", "input_audio"]
    text: Optional[str] = None
    image_url: Optional[Union[str, ImageURL]] = None
    input_audio: Optional[InputAudio] = None
    model_config = ConfigDict(extra="allow")


class Message(BaseModel):
    # Keep role permissive; don't over-validate (some providers also use "tool")
    role: str
    # Accept either a simple string or a list of content parts
    content: Union[str, List[ContentPart]]
    # Allow additional OpenAI fields (name, tool_call_id, tool_calls, etc.)
    model_config = ConfigDict(extra="allow")


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=32768, gt=0)
    stream: bool = False
    reasoning_effort: str = "high"

    # Be permissive with evolving OpenAI parameters (tools, response_format, etc.)
    model_config = ConfigDict(extra="allow")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _build_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    # Only attach Authorization if key is present; avoid "Bearer None"
    if WEBUI_SECRET_KEY:
        headers["Authorization"] = f"Bearer {WEBUI_SECRET_KEY}"
    return headers


def _flatten_text_only_content(messages: List[Message]) -> List[Dict[str, Any]]:
    """
    Some OpenAI-compatible backends still expect message.content to be a string.
    If a message has a list of *only* text parts, merge them into a string.
    Otherwise, keep the content as-is.
    """
    normalized: List[Dict[str, Any]] = []
    for m in messages:
        # dump everything except content
        base = m.model_dump(exclude={"content"}, exclude_none=True)
        content = m.content
        if isinstance(content, list):
            text_parts: List[str] = []
            non_text_found = False
            for part in content:
                if part.type == "text" and part.text is not None:
                    text_parts.append(part.text)
                else:
                    non_text_found = True
                    break
            if not non_text_found:
                # Join text parts with newlines (simple, lossless for text-only)
                base["content"] = "\n".join(text_parts)
            else:
                # Keep the structured content for backends that support it
                base["content"] = [p.model_dump(exclude_none=True) for p in content]
        else:
            base["content"] = content
        normalized.append(base)
    return normalized

# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------

@app.get("/ping")
async def ping():
    """Health check endpoint."""
    logger.info("Received /ping request.")
    # Respond with a minimal body so some health checkers don't treat an empty
    # body as a failure.
    return JSONResponse({"ok": True}, status_code=status.HTTP_200_OK)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, chat_request: ChatCompletionRequest):
    """
    Proxies chat completion requests to the OpenWebUI instance.
    Accepts both string and content-part list for message.content.
    """
    request_id = request.headers.get("X-Request-ID", "unknown")
    logger.info(f"[{request_id}] Received chat completion request for model: {chat_request.model}")

    headers = _build_headers()

    try:
        # Build a dict from the validated request. We keep it lean (exclude None).
        openai_payload: Dict[str, Any] = chat_request.model_dump(exclude_none=True)

        # Normalize messages so text-only content lists are flattened into strings
        openai_payload["messages"] = _flatten_text_only_content(chat_request.messages)

        if chat_request.stream:
            # --- Streaming path: keep the HTTPX stream open while yielding bytes ---
            async def iter_openwebui_stream():
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        OPENWEBUI_INTERNAL_API_URL,
                        json=openai_payload,
                        headers=headers,
                        timeout=None,
                    ) as r:
                        r.raise_for_status()
                        async for chunk in r.aiter_bytes():
                            yield chunk

            logger.info(f"[{request_id}] Streaming response from OpenWebUI.")
            # Upstreams typically use SSE
            return StreamingResponse(iter_openwebui_stream(), media_type="text/event-stream")

        else:
            # --- Non-streaming path ---
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    OPENWEBUI_INTERNAL_API_URL,
                    json=openai_payload,
                    headers=headers,
                    timeout=None,
                )
                response.raise_for_status()
                logger.info(f"[{request_id}] Non-streaming response from OpenWebUI.")
                return JSONResponse(content=response.json())

    # NOTE: Pydantic validation for the body happens *before* entering this function.
    # This except block is mainly for any additional internal validation you add later.
    except ValidationError as e:
        logger.error(f"[{request_id}] Request validation error: {e.errors()}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "Invalid request payload",
                    "type": "validation_error",
                    "code": "invalid_payload",
                    "errors": e.errors(),
                }
            },
        )

    except httpx.RequestError as e:
        logger.error(f"[{request_id}] Error proxying request to OpenWebUI: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Error communicating with OpenWebUI: {e}",
                    "type": "proxy_error",
                    "code": "openwebui_connection_error",
                }
            },
        )

    except httpx.HTTPStatusError as e:
        body_text = e.response.text
        logger.error(f"[{request_id}] OpenWebUI returned an error: {e.response.status_code} - {body_text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail={
                "error": {
                    "message": f"OpenWebUI returned an error: {body_text}",
                    "type": "openwebui_error",
                    "code": f"http_status_{e.response.status_code}",
                }
            },
        )

    except Exception as e:
        logger.error(f"[{request_id}] An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"An unexpected error occurred: {e}",
                    "type": "unexpected_error",
                    "code": "internal_server_error",
                }
            },
        )

# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 80))
    logger.info(f"Starting server on port {port}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
