import os
import re
import logging
import httpx
import uvicorn
import asyncio
import subprocess
import signal
import sys
from pathlib import Path
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
WEBUI_SECRET_KEY = os.getenv("WEBUI_SECRET_KEY", "rp-tutel-internal-key")

# Model readiness configuration
MODEL_STARTUP_TIMEOUT = int(os.getenv("MODEL_STARTUP_TIMEOUT", "600"))  # 10 minutes
MODEL_CHECK_INTERVAL = int(os.getenv("MODEL_CHECK_INTERVAL", "5"))  # 5 seconds

# ------------------------------------------------------------
# Global State
# ------------------------------------------------------------
model_ready = False
openwebui_process = None
startup_task = None

# ------------------------------------------------------------
# Model Management
# ------------------------------------------------------------

def ensure_model_symlink():
    """Ensure the model symlink exists."""
    openai_dir = Path("./openai")
    symlink_path = openai_dir / "gpt-oss-120b"
    target_path = Path("/runpod-volume/models/gpt-oss-120b")
    
    if not symlink_path.exists():
        logger.info(f"Creating symlink: {symlink_path} -> {target_path}")
        openai_dir.mkdir(parents=True, exist_ok=True)
        symlink_path.symlink_to(target_path)
    else:
        logger.info(f"Symlink already exists: {symlink_path}")


async def check_openwebui_health() -> bool:
    """Check if OpenWebUI is responding to health checks."""
    try:
        async with httpx.AsyncClient() as client:
            # First check if the service is up
            response = await client.get("http://localhost:8000", timeout=5.0)
            if response.status_code >= 500:
                return False
            
            # Try a simple completion to ensure model is actually ready
            test_payload = {
                "model": "gpt-oss-120b",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
                "stream": False
            }
            
            headers = _build_headers()
            test_response = await client.post(
                OPENWEBUI_INTERNAL_API_URL,
                json=test_payload,
                headers=headers,
                timeout=10.0
            )
            
            # If we get a valid response, model is ready
            return test_response.status_code == 200
            
    except Exception as e:
        # During startup, exceptions are expected
        return False


async def log_subprocess_output(process):
    """Read and log subprocess output in real-time."""
    try:
        while process.poll() is None:
            line = process.stdout.readline()
            if line:
                line = line.strip()
                # Log the output from OpenWebUI
                logger.info(f"[OpenWebUI] {line}")
                
                # Look for percentage indicators in the output
                # Common patterns: "Loading: 50%", "Progress: 50%", "[50%]", etc.
                percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
                if percent_match:
                    logger.info(f"[Model Loading Progress] {percent_match.group(0)}")
            else:
                await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"Error reading subprocess output: {e}")


async def start_openwebui_process():
    """Start the OpenWebUI process in the background."""
    global openwebui_process, model_ready
    
    # Ensure the model symlink exists
    ensure_model_symlink()
    
    # Start OpenWebUI process
    cmd = [
        "/opt/deepseek-tutel-accel/run.sh",
        "--serve=webui",
        "--listen_port", "8000",
        "--try_path", "./openai/gpt-oss-120b"
    ]
    
    logger.info(f"Starting OpenWebUI with command: {' '.join(cmd)}")
    
    try:
        openwebui_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=os.setsid  # Create new process group for better process management
        )
        logger.info(f"OpenWebUI process started with PID: {openwebui_process.pid}")
        
        # Start task to log subprocess output
        output_task = asyncio.create_task(log_subprocess_output(openwebui_process))
        
        # Wait for OpenWebUI to become ready
        elapsed = 0
        while elapsed < MODEL_STARTUP_TIMEOUT:
            # Check if process has died unexpectedly
            if openwebui_process.poll() is not None:
                logger.error(f"OpenWebUI process died unexpectedly with exit code: {openwebui_process.returncode}")
                # Read any remaining output
                remaining_output = openwebui_process.stdout.read()
                if remaining_output:
                    logger.error(f"Final output: {remaining_output}")
                return False
            
            if await check_openwebui_health():
                model_ready = True
                logger.info("OpenWebUI health check passed! Waiting 2 seconds for full initialization...")
                await asyncio.sleep(2)  # Give it a moment to fully stabilize
                logger.info("OpenWebUI is ready and accepting requests!")
                return True
            
            percentage = (elapsed / MODEL_STARTUP_TIMEOUT) * 100
            logger.info(f"Waiting for OpenWebUI to start... ({elapsed}s elapsed, {percentage:.1f}% of timeout)")
            await asyncio.sleep(MODEL_CHECK_INTERVAL)
            elapsed += MODEL_CHECK_INTERVAL
        
        logger.error(f"OpenWebUI did not start within {MODEL_STARTUP_TIMEOUT} seconds")
        output_task.cancel()
        return False
        
    except Exception as e:
        logger.error(f"Failed to start OpenWebUI: {e}", exc_info=True)
        return False


def shutdown_openwebui():
    """Gracefully shutdown the OpenWebUI process."""
    global openwebui_process
    
    if openwebui_process:
        logger.info(f"Shutting down OpenWebUI process (PID: {openwebui_process.pid})")
        try:
            # Try graceful termination first
            os.killpg(os.getpgid(openwebui_process.pid), signal.SIGTERM)
            openwebui_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("OpenWebUI did not terminate gracefully, forcing kill")
            try:
                os.killpg(os.getpgid(openwebui_process.pid), signal.SIGKILL)
            except:
                openwebui_process.kill()
            openwebui_process.wait()
        except Exception as e:
            logger.error(f"Error shutting down OpenWebUI: {e}")
            try:
                openwebui_process.kill()
                openwebui_process.wait()
            except:
                pass
        finally:
            openwebui_process = None


# ------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the FastAPI application."""
    global startup_task
    
    logger.info("FastAPI application startup.")
    
    # Start the OpenWebUI process asynchronously
    startup_task = asyncio.create_task(start_openwebui_process())
    
    # Don't wait for it to complete - let the server start immediately
    # The task will run in the background
    
    yield
    
    logger.info("FastAPI application shutdown.")
    
    # Cancel startup task if still running
    if startup_task and not startup_task.done():
        startup_task.cancel()
        try:
            await startup_task
        except asyncio.CancelledError:
            pass
    
    # Shutdown OpenWebUI process
    shutdown_openwebui()


app = FastAPI(title="OpenWebUI Proxy Load Balancer", version="1.0.0", lifespan=lifespan)

# ------------------------------------------------------------
# Pydantic Models (OpenAI-compatible and permissive)
# ------------------------------------------------------------

class ImageURL(BaseModel):
    url: str
    detail: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class InputAudio(BaseModel):
    data: str
    format: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class ContentPart(BaseModel):
    type: Literal["text", "image_url", "input_audio"]
    text: Optional[str] = None
    image_url: Optional[Union[str, ImageURL]] = None
    input_audio: Optional[InputAudio] = None
    model_config = ConfigDict(extra="allow")


class Message(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]
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
    model_config = ConfigDict(extra="allow")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _build_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
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
                base["content"] = "\n".join(text_parts)
            else:
                base["content"] = [p.model_dump(exclude_none=True) for p in content]
        else:
            base["content"] = content
        normalized.append(base)
    return normalized

# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------

@app.get("/health")
async def health():
    """
    Health check endpoint that indicates both service and model status.
    Returns 200 if service is up, with model_ready flag in response.
    """
    return JSONResponse({
        "status": "healthy",
        "model_ready": model_ready,
        "service": "running"
    }, status_code=status.HTTP_200_OK)


@app.get("/ping")
async def ping():
    """Simple liveness check endpoint."""
    logger.info("Received /ping request.")
    return JSONResponse({"ok": True}, status_code=status.HTTP_200_OK)


@app.get("/ready")
async def ready():
    """
    Readiness check endpoint.
    Returns 200 if model is ready, 503 if still loading.
    """
    if model_ready:
        return JSONResponse({"ready": True}, status_code=status.HTTP_200_OK)
    else:
        return JSONResponse(
            {"ready": False, "message": "Model is still loading"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, chat_request: ChatCompletionRequest):
    """
    Proxies chat completion requests to the OpenWebUI instance.
    Returns 503 if model is not ready yet.
    """
    request_id = request.headers.get("X-Request-ID", "unknown")
    logger.info(f"[{request_id}] Received chat completion request for model: {chat_request.model}")
    
    # Check if model is ready
    if not model_ready:
        logger.warning(f"[{request_id}] Model not ready yet, returning 503")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "message": "Model is still loading. Please try again in a few moments.",
                    "type": "model_not_ready",
                    "code": "model_loading",
                }
            }
        )

    headers = _build_headers()

    try:
        openai_payload: Dict[str, Any] = chat_request.model_dump(exclude_none=True)
        openai_payload["messages"] = _flatten_text_only_content(chat_request.messages)

        if chat_request.stream:
            async def iter_openwebui_stream():
                try:
                    async with httpx.AsyncClient() as client:
                        # Add retry logic for streaming requests
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                async with client.stream(
                                    "POST",
                                    OPENWEBUI_INTERNAL_API_URL,
                                    json=openai_payload,
                                    headers=headers,
                                    timeout=httpx.Timeout(60.0, connect=10.0),  # More specific timeout
                                ) as r:
                                    r.raise_for_status()
                                    async for chunk in r.aiter_bytes():
                                        if chunk:  # Only yield non-empty chunks
                                            yield chunk
                                break  # Success, exit retry loop
                                
                            except (httpx.ReadError, httpx.RemoteProtocolError) as e:
                                if attempt < max_retries - 1:
                                    logger.warning(f"[{request_id}] Streaming error (attempt {attempt + 1}/{max_retries}): {e}")
                                    await asyncio.sleep(0.5)  # Brief delay before retry
                                else:
                                    logger.error(f"[{request_id}] Streaming failed after {max_retries} attempts: {e}")
                                    raise
                                    
                except Exception as e:
                    logger.error(f"[{request_id}] Streaming error: {e}")
                    # Send an error message in SSE format
                    error_msg = f"data: {{\"error\": \"Streaming failed: {str(e)}\"}}\n\n"
                    yield error_msg.encode()

            logger.info(f"[{request_id}] Streaming response from OpenWebUI.")
            return StreamingResponse(iter_openwebui_stream(), media_type="text/event-stream")

        else:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                response = await client.post(
                    OPENWEBUI_INTERNAL_API_URL,
                    json=openai_payload,
                    headers=headers,
                )
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
# Signal Handlers
# ------------------------------------------------------------

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {sig}, shutting down...")
    shutdown_openwebui()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

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
