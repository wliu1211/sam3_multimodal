import asyncio
import time
import torch
from fastapi import HTTPException
from functools import wraps
from middleware.logger import get_logger

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

# How long a request may wait in the queue before being rejected with 503.
# Set to slightly less than the frontend's 120s timeout so the client receives
# a meaningful error message rather than a raw ECONNABORTED.
QUEUE_WAIT_TIMEOUT_S = 100

# ── Global GPU lock ───────────────────────────────────────────────────────────

# A single asyncio.Lock that every route acquires before touching the GPU.
# This serializes all vision pipeline calls so SAM 3 and CLIP are never
# called concurrently — concurrent GPU calls would either crash with an
# out-of-memory error or produce corrupted results.
#
# For a single-user demo this is the right primitive.
# For multi-user production, replace with a proper task queue (Celery + Redis).
_gpu_lock = asyncio.Lock()


# ── Context manager ───────────────────────────────────────────────────────────

class GpuQueue:
    """
    Async context manager that wraps the global GPU lock.

    - Waits at most QUEUE_WAIT_TIMEOUT_S seconds to acquire the lock;
      raises HTTP 503 if the queue is too backed-up so the client gets a
      useful message instead of a raw connection-aborted timeout.
    - Flushes the CUDA memory cache on every release so that cached-but-freed
      GPU memory is returned promptly, keeping per-inference allocation time
      stable across many successive calls.

    Usage in a route:
        async with GpuQueue():
            instances = await run_vision_pipeline(...)
    """

    async def __aenter__(self):
        if _gpu_lock.locked():
            logger.info("[queue] GPU is busy — request queued, waiting...")

        t0 = time.monotonic()
        try:
            await asyncio.wait_for(
                _gpu_lock.acquire(),
                timeout=QUEUE_WAIT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            waited = time.monotonic() - t0
            logger.warning(
                f"[queue] Request rejected after waiting {waited:.1f}s — "
                "GPU queue is overloaded."
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "The server is busy processing another request. "
                    "Please wait a moment and try again."
                ),
            )

        waited = time.monotonic() - t0
        if waited > 1.0:
            logger.info(f"[queue] GPU lock acquired after {waited:.1f}s wait")
        else:
            logger.debug("[queue] GPU lock acquired")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        _gpu_lock.release()
        logger.debug("[queue] GPU lock released")

        # Flush the CUDA allocator cache so freed tensors are returned to the
        # device memory pool immediately.  Without this, the cache grows with
        # every inference call and eventually causes slower allocations or OOM.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Don't suppress exceptions — let them propagate normally
        return False


# ── Decorator (optional convenience) ─────────────────────────────────────────

def gpu_queued(func):
    """
    Decorator version of GpuQueue for async functions that should be
    serialized through the GPU lock.

    Usage:
        @gpu_queued
        async def run_vision_pipeline(...):
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        async with GpuQueue():
            return await func(*args, **kwargs)
    return wrapper