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
# The tutel script exposes an OpenAI-compatible API at /v1/chat/completions
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "8000"))
TUTEL_INTERNAL_API_URL = f"http://localhost:{LISTEN_PORT}/v1/chat/completions"

# Retrieve the internal API key set in the entrypoint.sh
WEBUI_SECRET_KEY = os.getenv("WEBUI_SECRET_KEY", "rp-tutel-internal-key")

# Model readiness configuration
MODEL_STARTUP_TIMEOUT = int(os.getenv("MODEL_STARTUP_TIMEOUT", "600"))  # 10 minutes
MODEL_CHECK_INTERVAL = int(os.getenv("MODEL_CHECK_INTERVAL", "5"))  # 5 seconds

# Tutel configuration from env vars
HF_MODEL = os.getenv("HF_MODEL", "openai/gpt-oss-120b")
PATH_TO_MODEL = os.getenv("PATH_TO_MODEL", "/data/models/gpt-oss-120b")
MAX_SEQ_LEN = os.getenv("MAX_SEQ_LEN", "131072")
BUFFER_SIZE = os.getenv("BUFFER_SIZE", "256")
SERVE = os.getenv("SERVE", "core")
PROMPT = os.getenv("PROMPT", "")
DISABLE_THINKING = os.getenv("DISABLE_THINKING", "false").lower() in ("true", "1", "yes")
DISABLE_FP4 = os.getenv("DISABLE_FP4", "false").lower() in ("true", "1", "yes")

# ------------------------------------------------------------
# Global State
# ------------------------------------------------------------
model_ready = False
openwebui_process = None
startup_task = None

async def check_openwebui_health() -> bool:
    """Check if OpenWebUI is responding to health checks."""
    try:
        async with httpx.AsyncClient() as client:
            # First check if the service is up
            response = await client.get(f"http://localhost:{LISTEN_PORT}", timeout=5.0)
            if response.status_code >= 500:
                return False
            
            # Try a simple completion to ensure model is actually ready
            test_payload = {
                "model": HF_MODEL,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
                "stream": False
            }
            
            headers = _build_headers()
            test_response = await client.post(
                TUTEL_INTERNAL_API_URL,
                json=test_payload,
                headers=headers,
                timeout=10.0
            )
            
            # If we get a valid response, model is ready
            return test_response.status_code == 200
            
    except Exception:
        # During startup, exceptions are expected
        return False


async def log_subprocess_output_improved(process):
    """
    Improved subprocess output reader that handles both stdout and stderr
    with better buffering and real-time streaming.
    """
    try:
        # Create a queue for output lines
        output_queue = asyncio.Queue()
        
        async def read_stream(stream, stream_name):
            """Read from a stream and put lines into the queue."""
            try:
                while True:
                    # Read line asynchronously
                    line = await stream.readline()
                    if not line:
                        break
                    
                    # Decode and strip the line
                    decoded = line.decode('utf-8', errors='replace').strip()
                    if decoded:
                        await output_queue.put((stream_name, decoded))
            except Exception as e:
                logger.error(f"Error reading {stream_name}: {e}")
            finally:
                await output_queue.put((stream_name, None))  # Signal completion
        
        # Start readers for both stdout and stderr
        stdout_task = asyncio.create_task(read_stream(process.stdout, 'stdout'))
        stderr_task = asyncio.create_task(read_stream(process.stderr, 'stderr'))
        
        streams_done = {'stdout': False, 'stderr': False}
        
        # Process output from the queue
        while not all(streams_done.values()):
            try:
                # Wait for output with timeout
                stream_name, line = await asyncio.wait_for(
                    output_queue.get(),
                    timeout=1.0
                )
                
                if line is None:
                    streams_done[stream_name] = True
                    continue

                # Look for percentage indicators in the output
                percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
                if percent_match and "oading" in line:
                    logger.info(f"[Model Loading Progress] {percent_match.group(0)}")
                else:
                    logger.info(f"[Tutel] {line}")
                
                # Look for specific markers
                if "Model ready!" in line or "Start listening on" in line:
                    logger.info("[Tutel] *** MODEL IS READY ***")
                
            except asyncio.TimeoutError:
                # No output for 1 second, continue waiting
                continue
                
        # Wait for tasks to complete
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        
    except Exception as e:
        logger.error(f"Error in output reader: {e}")


async def start_openwebui_process():
    """Start Tutel core server with improved subprocess handling."""
    global openwebui_process, model_ready
    
    python_bin = sys.executable
    cmd = [
        python_bin, "-u", "/opt/deepseek-tutel-accel/llm_moe_tutel.py",
    ]
    
    cmd.extend(["--serve", SERVE])
    cmd.extend(["--listen_port", str(LISTEN_PORT)])
    cmd.extend(["--hf_model", HF_MODEL])
    cmd.extend(["--path_to_model", PATH_TO_MODEL])
    cmd.extend(["--max_seq_len", MAX_SEQ_LEN])
    cmd.extend(["--buffer_size", BUFFER_SIZE])
    if PROMPT:
        cmd.extend(["--prompt", PROMPT])
    if DISABLE_THINKING:
        cmd.extend(["--disable_thinking"])
    if DISABLE_FP4:
        cmd.extend(["--disable_fp4"])
    
    logger.info(f"Starting Tutel core with command: {' '.join(cmd)}")
    
    try:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + ":/usr/lib/x86_64-linux-gnu"
        env.setdefault("NCCL_DEBUG", "INFO")
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")  # Force unbuffered Python output
        env.setdefault("HF_HUB_OFFLINE", "0")
        env.setdefault("LOCAL_SIZE", os.getenv("LOCAL_SIZE", "1"))
        
        # Create subprocess with async pipes
        openwebui_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            preexec_fn=os.setsid,  # Create new process group
            limit=1024*1024  # 1MB buffer limit
        )
        
        logger.info(f"Tutel process started with PID: {openwebui_process.pid}")
        
        # Start output reader task
        output_task = asyncio.create_task(log_subprocess_output_improved(openwebui_process))
        
        # Wait for model to be ready
        elapsed = 0
        last_health_check = 0
        
        while elapsed < MODEL_STARTUP_TIMEOUT:
            # Check if process has died
            if openwebui_process.returncode is not None:
                logger.error(f"Tutel process died with exit code: {openwebui_process.returncode}")
                
                # Try to get any remaining output
                try:
                    await asyncio.wait_for(output_task, timeout=2.0)
                except asyncio.TimeoutError:
                    pass
                
                return False
            
            # Perform health check every MODEL_CHECK_INTERVAL seconds
            current_time = elapsed
            if current_time - last_health_check >= MODEL_CHECK_INTERVAL:
                if await check_openwebui_health():
                    model_ready = True
                    logger.info("Health check passed; model is ready!")
                    
                    # Let output reader continue running in background
                    return True
                last_health_check = current_time
            
            # Progress indicator
            percentage = (elapsed / MODEL_STARTUP_TIMEOUT) * 100
            if elapsed % 10 == 0:  # Log every 10 seconds
                logger.info(f"Waiting for Tutel… ({elapsed}s elapsed, {percentage:.1f}% of timeout)")
            
            await asyncio.sleep(1)
            elapsed += 1
        
        logger.error(f"Tutel did not start within {MODEL_STARTUP_TIMEOUT} seconds")
        
        # Cancel output task
        output_task.cancel()
        try:
            await output_task
        except asyncio.CancelledError:
            pass
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to start Tutel: {e}", exc_info=True)
        return False


def shutdown_openwebui():
    """Gracefully shutdown the OpenWebUI process."""
    global openwebui_process
    
    if openwebui_process:
        logger.info(f"Shutting down OpenWebUI process (PID: {openwebui_process.pid})")
        try:
            # Try graceful termination first
            os.killpg(os.getpgid(openwebui_process.pid), signal.SIGTERM)
            
            # Wait for process to terminate (async wait in sync context)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context, create a task
                    asyncio.create_task(openwebui_process.wait())
                else:
                    # If not, run it synchronously
                    loop.run_until_complete(
                        asyncio.wait_for(openwebui_process.wait(), timeout=10)
                    )
            except (asyncio.TimeoutError, RuntimeError):
                logger.warning("OpenWebUI did not terminate gracefully, forcing kill")
                try:
                    os.killpg(os.getpgid(openwebui_process.pid), signal.SIGKILL)
                except:
                    openwebui_process.kill()
                
                # Wait for forced termination
                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_running():
                        loop.run_until_complete(openwebui_process.wait())
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error shutting down OpenWebUI: {e}")
            try:
                openwebui_process.kill()
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
    temperature: float = Field(default=float(os.getenv("TEMP", "1.0")), ge=0.0, le=2.0)
    top_p: float = Field(default=float(os.getenv("TOP_P", "1.0")), ge=0.0, le=1.0)
    top_k: int = Field(default=int(os.getenv("TOP_K", "0")), ge=0)
    min_p: float = Field(default=float(os.getenv("MIN_P", "0.0")), ge=0.0, le=1.0)
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


def _flatten_content_parts_to_text(content: Union[str, List[Any]]) -> str:
    """
    Convert OpenAI-style content (string or list of parts) to a plain text string.
    - If already a string, return as-is.
    - If a list, join 'text' parts; for non-text parts, include a lightweight placeholder.
    """
    if isinstance(content, str):
        return content

    parts_out: List[str] = []
    for part in content:
        # part may be a dict (from model_dump) or a ContentPart
        p_type = None
        text = None
        image_url = None

        if isinstance(part, dict):
            p_type = part.get("type")
            text = part.get("text")
            image_url = part.get("image_url")
        else:
            # Pydantic model instance
            p_type = getattr(part, "type", None)
            text = getattr(part, "text", None)
            image_url = getattr(part, "image_url", None)

        if p_type == "text" and text:
            parts_out.append(text)
        elif p_type == "image_url" and image_url:
            # image_url can be str or {"url": "..."}
            url_str = None
            if isinstance(image_url, str):
                url_str = image_url
            elif isinstance(image_url, dict):
                url_str = image_url.get("url")
            else:
                url_str = getattr(image_url, "url", None)
            if url_str:
                parts_out.append(f"[image: {url_str}]")
        elif p_type == "input_audio":
            parts_out.append("[audio]")
        # silently ignore unknown types

    return "\n".join([p for p in parts_out if p]).strip()


def _normalize_for_openwebui(chat_request: ChatCompletionRequest) -> Dict[str, Any]:
    """
    Build the payload for OpenWebUI/Tutel, flattening list-style 'content' into strings.
    """
    payload: Dict[str, Any] = chat_request.model_dump(exclude_none=True)

    normalized_messages: List[Dict[str, Any]] = []
    for m in payload.get("messages", []):
        role = m.get("role")
        content = m.get("content")
        flattened = _flatten_content_parts_to_text(content)
        normalized_messages.append({"role": role, "content": flattened})

    payload["messages"] = normalized_messages
    return payload

# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------

@app.get("/ping")
async def ping():
    """
    Readiness check endpoint.
    Returns 200 if model is ready, 204 if still loading.
    """    
    if model_ready:
        return JSONResponse({"status": "healthy"}, status_code=status.HTTP_200_OK)
    else:
        logger.info("Received /ping request while model still loading.")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, chat_request: ChatCompletionRequest):
    """
    Proxies chat completion requests to the OpenWebUI instance.
    Returns 503 if model is not ready yet after waiting.
    """
    request_id = request.headers.get("X-Request-ID", "unknown")
    logger.info(f"[{request_id}] Received chat completion request for model: {chat_request.model}")
   
    # Check if model is ready, wait up to 5 minutes
    wait_timeout = 300  # 5 minutes
    check_interval = 5  # seconds
    elapsed = 0
    while not model_ready and elapsed < wait_timeout:
        logger.info(f"[{request_id}] Model not ready, waiting... ({elapsed}s elapsed)")
        await asyncio.sleep(check_interval)
        elapsed += check_interval
   
    if not model_ready:
        logger.warning(f"[{request_id}] Model not ready after {wait_timeout}s, returning 503")
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
        # NORMALIZE: flatten list content into plain text for OpenWebUI/Tutel
        openai_payload: Dict[str, Any] = _normalize_for_openwebui(chat_request)

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
                                    TUTEL_INTERNAL_API_URL,
                                    json=openai_payload,
                                    headers=headers,
                                    timeout=httpx.Timeout(600.0, connect=10.0),  # More specific timeout
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
            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
                response = await client.post(
                    TUTEL_INTERNAL_API_URL,
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
